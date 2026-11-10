"""Hero route: a full-body model with a high-detail head grafted on, as pure functions.

Pixal3D spends its detail on its whole box, so a face that is a few dozen pixels of a
full-body picture comes out soft (goggle eyes, smeared paint). The same face cropped,
upscaled and built on its own fills the box and keeps real eyelids. This module holds the
maths that puts that head back on the body; `scripts/hero_route.py` runs the steps.

The head is found from a throwaway rig of the body: the topmost bone region, plus any bone
whose middle sits inside the head's height (the neck when it carries hair or a collar).
That needs no names and no hand-drawn box, and matched the hand-picked heads on both
characters it was tried on (warrior: one bone; elf: head + neck).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

LIGHT_GREY = 236          # the studio's plain background, for pictures without a cut-out
GREY_TOLERANCE = 25
CROP_PAD = 1.3            # crop side vs the head's own size, so hair and ears fit
CROP_DROP = 0.05          # shift the crop down a little: chin and neck matter more than air
HEAD_FLOOR_PERCENTILE = 5
OUTLINE_GROW_PX = 4       # the least margin round the real head's outline
OUTLINE_GROW_SHARE = 0.1  # else a tenth of the head's width
LINER_SHRINK = 0.04       # the body's own head, kept as a liner under the new hair, shrinks this much
LINER_TUCK = 0.97         # a liner vertex outside the new head pulls back to this of its surface
LINER_REACH = 1.5         # liner work stops this many skull radii from the neck's centre line
SKULL_SHARE = 0.55        # skull fit: the top this share of each head's height
COLLAR_RATIO = 1.15       # a slice this much wider than the neck above it is a collar, not more neck
MIN_SLICE_POINTS = 20     # fewer points than this in a slice and its width is noise

_COMPONENTS = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def picture_box(image: Image.Image) -> tuple[int, int, int, int]:
    """The figure's box (x0, x1, y0, y1). A cut-out (alpha) wins; else anything far from
    the studio's light grey, which misreads darker or graded backgrounds as figure."""
    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if alpha.min() < 128:
            ys, xs = np.nonzero(alpha > 127)
            return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    rgb = np.asarray(image.convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(rgb - LIGHT_GREY).max(-1) > GREY_TOLERANCE)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def frame(vertices: np.ndarray, box: tuple[int, int, int, int]):
    """(mesh centre, world units per pixel, box centre). Pixal3D fits every model to the
    same box, so a model's scale against its picture comes from the two heights."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    x0, x1, y0, y1 = box
    return (lo + hi) / 2, (hi[1] - lo[1]) / (y1 - y0), np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def project(points: np.ndarray, fr) -> tuple[np.ndarray, np.ndarray]:
    """Front-view picture pixels (x, y) of world points; picture y runs down."""
    centre, per_px, box_centre = fr
    return (box_centre[0] + (points[:, 0] - centre[0]) / per_px,
            box_centre[1] - (points[:, 1] - centre[1]) / per_px)


def _accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    dtype, width = _COMPONENTS[accessor["componentType"]], _WIDTH[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    item = np.dtype(dtype).itemsize * width
    stride = view.get("byteStride", item)
    rows = np.lib.stride_tricks.as_strided(
        np.frombuffer(binary, dtype=np.uint8, offset=start),
        shape=(accessor["count"], stride), strides=(stride, 1))
    return rows[:, :item].copy().view(dtype).reshape(-1, width)


def skinned_vertices(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Positions and dominant joint of every vertex of every skinned primitive in a GLB."""
    length = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20:20 + length])
    binary = glb[20 + length + 8:]
    positions, joints = [], []
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "JOINTS_0" not in attributes:
                continue
            positions.append(_accessor(document, binary, attributes["POSITION"]))
            index = _accessor(document, binary, attributes["JOINTS_0"])
            weight = _accessor(document, binary, attributes["WEIGHTS_0"])
            joints.append(index[np.arange(len(index)), weight.argmax(axis=1)])
    if not positions:
        raise ValueError("the rig has no skinned mesh")
    return np.concatenate(positions), np.concatenate(joints).astype(int)


def boxed(points: np.ndarray) -> np.ndarray:
    """Points centred and scaled to unit height, so two exports of one model line up."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    return (points - (lo + hi) / 2) / (hi[1] - lo[1])


def regions(vertices: np.ndarray, rig_positions: np.ndarray, rig_joints: np.ndarray) -> np.ndarray:
    """The bone of every vertex: the nearest rigged vertex once both are boxed to one height
    (the rigger may re-export the mesh at another scale or orientation)."""
    _, nearest = cKDTree(boxed(rig_positions)).query(boxed(vertices), workers=-1)
    return rig_joints[nearest]


def head_bones(bone: np.ndarray, heights: np.ndarray) -> list[int]:
    """The topmost bone region, plus every bone whose median height sits inside it."""
    bones = np.unique(bone)
    middle = {int(b): float(np.median(heights[bone == b])) for b in bones}
    top = max(middle, key=middle.get)
    floor = np.percentile(heights[bone == top], HEAD_FLOOR_PERCENTILE)
    return sorted(b for b, m in middle.items() if m >= floor)


def neck_height(heights: np.ndarray, xs: np.ndarray, slices: int = 30) -> float | None:
    """Where to cut the head off: the top of a collar, the first slice above it that is clearly
    narrower (the jaw).

    The rig's head region can reach down into a collar (the warrior's did), and the new head,
    built from a crop whose edge runs through that collar, makes a scuffed one. Cutting at the
    collar's top keeps the body's own collar. Not the narrowest slice: the neck hides inside the
    collar, so the narrow band is the chin, and cutting through it left the old mouth showing.
    None when nothing below is clearly wider (a bare neck): swap the whole region.
    """
    edges = np.linspace(heights.min(), heights.max(), slices + 1)
    widths = np.full(slices, np.nan)
    for i in range(slices):
        inside = (heights >= edges[i]) & (heights < edges[i + 1])
        if inside.sum() >= MIN_SLICE_POINTS:
            widths[i] = np.percentile(xs[inside], 98) - np.percentile(xs[inside], 2)
    valid = np.flatnonzero(~np.isnan(widths))
    lower, upper = valid[valid < slices // 2], valid[valid >= slices // 2]
    if not len(lower) or not len(upper):
        return None
    collar = lower[np.argmax(widths[lower])]
    skull = upper[np.argmax(widths[upper])]
    between = valid[(valid > collar) & (valid < skull)]
    if not len(between):
        return None
    narrower = between[widths[collar] >= COLLAR_RATIO * widths[between]]
    return float(edges[narrower[0]]) if len(narrower) else None


NECK_SLACK = 1.12          # the collar's top: the first slice from below within this much of the neck's narrowest
NECK_MARGIN = 0.03        # then this share of the head's height higher, clear of the collar's rim


def collar_top(vertices: np.ndarray, real: np.ndarray) -> float:
    """The height of the collar's top on the body, from its width profile (up = y).

    From a little below the head region up to just under the mouth, slice the body near the neck's
    centre line (arms left out); the neck's narrowest slice sets the scale and the first slice from
    below that is within NECK_SLACK of it is where the collar ends. `neck_height` looks inside the
    head region only and on a slim neck finds the narrow band under the jaw instead, which put the
    cut through the chin. A bare neck has no collar: its lowest neck slice is returned."""
    heights, xs = vertices[:, 1], vertices[:, 0]
    lo, hh = float(heights[real].min()), float(np.ptp(heights[real]))
    axis = float(np.median(xs[real]))
    near = np.abs(xs - axis) < 0.6 * float(np.ptp(xs[real]))
    step = 0.01 * hh
    zs = np.arange(lo - 0.15 * hh, lo + 0.45 * hh, step)
    widths = np.full(len(zs), np.nan)
    for i, z in enumerate(zs):
        sl = near & (np.abs(heights - z) < step)
        if sl.sum() >= MIN_SLICE_POINTS:
            widths[i] = np.percentile(xs[sl], 98) - np.percentile(xs[sl], 2)
    if np.isnan(widths).all():
        return lo
    neck = np.nanmin(widths)
    found = np.flatnonzero(widths <= NECK_SLACK * neck)
    return float(zs[found[0]] + NECK_MARGIN * hh)


CHIN_JUMP = 0.15          # the chin starts where the front of a slice juts this share of the head's height
CHIN_CLEAR = 0.02         # past the neck below it; the collar cut goes this share of the head height lower


def chin_bottom(vertices: np.ndarray, real: np.ndarray) -> float | None:
    """The height where the body's jaw starts (up = y, front = +z): going up the neck, the first
    slice whose front juts CHIN_JUMP of the head's height past the furthest-back front below it.
    On a short neck the jaw hangs below the collar's top, and a cut at the collar left the old
    chin under the new head. None if no slice juts out."""
    heights = vertices[:, 1]
    lo, hh = float(heights[real].min()), float(np.ptp(heights[real]))
    axis = np.median(vertices[real][:, [0, 2]], axis=0)
    near = np.hypot(vertices[:, 0] - axis[0], vertices[:, 2] - axis[1]) < 0.45 * hh
    step = 0.01 * hh
    back_most = None
    for z in np.arange(lo - 0.15 * hh, lo + 0.45 * hh, step):
        sl = near & (np.abs(heights - z) < step)
        if sl.sum() < MIN_SLICE_POINTS:
            continue
        front = float(np.percentile(vertices[sl, 2], 98))
        if back_most is not None and front - back_most > CHIN_JUMP * hh:
            return float(z)
        back_most = front if back_most is None else min(back_most, front)
    return None


def neck_ring(vertices: np.ndarray, real: np.ndarray, cut: float) -> tuple[np.ndarray, float]:
    """(the neck's centre line as x, z; the skull's radius round it), from the upper half of the
    head above CUT. A neck is narrower than the skull and a collar wider, so the skull's radius
    is the ring between them, whatever the paint (no skin-colour guess)."""
    y = vertices[:, 1]
    skull = real & (y >= cut + 0.5 * (y[real].max() - cut))
    axis = np.median(vertices[skull][:, [0, 2]], axis=0)
    radius = np.hypot(vertices[skull, 0] - axis[0], vertices[skull, 2] - axis[1])
    return axis, float(np.percentile(radius, 90))


NECK_RING_SLACK = 1.12    # the neck/collar split: this much outside the neck's own radius


def neck_core(vertices: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, float]:
    """(the neck's centre line as x, z; the split radius between neck and collar), from the body's
    narrowest neck slice: its own centre (the neck sits behind the skull's) and NECK_RING_SLACK times
    its radius. For a tight collar, narrower than the skull, where `neck_ring` (the skull's radius)
    calls the whole collar neck."""
    heights, xs = vertices[:, 1], vertices[:, 0]
    lo, hh = float(heights[real].min()), float(np.ptp(heights[real]))
    axis0 = float(np.median(xs[real]))
    near = np.abs(xs - axis0) < 0.6 * float(np.ptp(xs[real]))
    step = 0.01 * hh
    best = None
    for z in np.arange(lo - 0.15 * hh, lo + 0.45 * hh, step):
        sl = near & (np.abs(heights - z) < step)
        if sl.sum() < MIN_SLICE_POINTS:
            continue
        w = np.percentile(xs[sl], 98) - np.percentile(xs[sl], 2)
        if best is None or w < best[0]:
            best = (w, sl)
    if best is None:
        return np.median(vertices[real][:, [0, 2]], axis=0), float(np.ptp(xs[real]))
    pts = vertices[best[1]][:, [0, 2]]
    centre = (np.percentile(pts, 2, axis=0) + np.percentile(pts, 98, axis=0)) / 2
    radius = np.percentile(np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]), 90)
    return centre, float(NECK_RING_SLACK * radius)


def head_cut(body: np.ndarray, real: np.ndarray, head: np.ndarray, cut: float | None,
             axis: np.ndarray, ring: float) -> tuple[np.ndarray, np.ndarray]:
    """(which body vertices the new head replaces, which new-head vertices may stay).

    Above CUT (the collar's top) the whole head swaps. Below it, in the collar band, a height
    cannot split them: the new head can sit a little lower, so its mouth shares heights with the
    body's old chin and both sit inside the collar. So the band splits by distance from the neck's
    centre line: inside RING is head (the body's old chin goes, the new chin stays), outside is
    collar (the body's stays, the new head's scuffed one goes). CUT None swaps the whole region."""
    if cut is None:
        return real.copy(), np.ones(len(head), bool)

    def inside(points):
        return np.hypot(points[:, 0] - axis[0], points[:, 2] - axis[1]) < ring

    return real & ((body[:, 1] >= cut) | inside(body)), (head[:, 1] >= cut) | inside(head)


def crop_box(px: np.ndarray, py: np.ndarray, width: int, height: int,
             pad: float = CROP_PAD, drop: float = CROP_DROP) -> tuple[int, int, int, int]:
    """A square crop (x0, y0, x1, y1) round the head's pixels, padded and kept in the picture."""
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    side = min(max(x1 - x0, y1 - y0) * pad, width, height)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + side * drop
    left = int(round(np.clip(cx - side / 2, 0, width - side)))
    top = int(round(np.clip(cy - side / 2, 0, height - side)))
    side = int(round(side))
    return left, top, left + side, top + side


def crop_to_full(full_frame, head_frame, crop: tuple[int, int, int, int], head_width: int) -> np.ndarray:
    """Transform taking the head model (framed by its crop picture) into the full model's frame:
    a head point at crop pixel p sits at full pixel crop0 + p * (crop side / head picture side)."""
    c_f, s_f, b_f = full_frame
    c_h, s_h, b_h = head_frame
    r = (crop[2] - crop[0]) / head_width
    k = s_f * r / s_h
    shift = c_f - c_h * k
    shift[0] += (crop[0] + b_h[0] * r - b_f[0]) * s_f
    shift[1] -= (crop[1] + b_h[1] * r - b_f[1]) * s_f
    tf = np.diag([k, k, k, 1.0])
    tf[:3, 3] = shift
    return tf


def place_on(real: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Transform putting NEW on REAL: the same centre and the same height (no other scale guess)."""
    lo_r, hi_r = real.min(axis=0), real.max(axis=0)
    lo_n, hi_n = new.min(axis=0), new.max(axis=0)
    k = (hi_r[1] - lo_r[1]) / (hi_n[1] - lo_n[1])
    tf = np.diag([k, k, k, 1.0])
    tf[:3, 3] = (lo_r + hi_r) / 2 - k * (lo_n + hi_n) / 2
    return tf


def silhouette(faces: np.ndarray, px: np.ndarray, py: np.ndarray, shape: tuple[int, int],
               grow: int = 0) -> np.ndarray:
    """Front-view outline of FACES as a filled mask. Whole triangles, not corner points:
    corners alone leave gaps on a low-poly mesh, and anything over a gap was dropped as a hole."""
    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for a, b, c in faces:
        draw.polygon([(px[a], py[a]), (px[b], py[b]), (px[c], py[c])], fill=255)
    mask = ndimage.binary_fill_holes(np.asarray(canvas) > 0)
    return ndimage.binary_dilation(mask, iterations=grow) if grow else mask


def on_mask(px: np.ndarray, py: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Which points land on MASK pixels."""
    x, y = np.round(px).astype(int), np.round(py).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    hit = np.zeros(len(x), bool)
    hit[ok] = mask[y[ok], x[ok]]
    return hit


def liner_split(vertices: np.ndarray, real: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices with the old head shrunk, which vertices go). The old head's back half stays
    as a liner under the new hair, shrunk about its centre by SHRINK.

    Thin hair layers thin out further when cut down and have nothing beneath, so gaps showed
    air. The old head is solid and painted from the same picture (its back from the back
    picture), so a gap now shows hair. Only the back half: the new face sits a little behind
    the old one, and the old face pushed through it. Models face +Z. SHRINK 0 drops it all.
    """
    if not shrink:
        return vertices, real.copy()
    moved = vertices.copy()
    inner = np.flatnonzero(real)
    centre = vertices[inner].mean(axis=0)
    moved[inner] = centre + (vertices[inner] - centre) * (1 - shrink)
    return moved, real & (vertices[:, 2] > centre[2])


def tuck_liner(vertices: np.ndarray, faces: np.ndarray, liner: np.ndarray, head, cut: float,
               axis: np.ndarray, radius: float, keep: float = LINER_TUCK,
               reach: float = LINER_REACH) -> tuple[np.ndarray, np.ndarray]:
    """(vertices with the old head's liner pulled inside HEAD, which faces to drop).

    The liner is the body's own head, kept under the new hair. An old head with more volume than
    the new one -- hair, or ears the new head does not have -- pokes out beside the new ears as a
    second set of them (the space marine, 2026-11-09). So: a ray from the new head's centre out
    through each liner vertex. A vertex outside the head is pulled to KEEP of the NEAREST hit.
    The nearest hit is the skull; the farthest is often an ear, which parks old skin just inside
    the ear and still shows.

    Only beside the head: above CUT, within REACH skull radii of the neck's centre line (AXIS as
    x, z; RADIUS the skull's), and further sideways than behind. Below those are shoulders and
    collar, and tucking them pulled the collar up into a cape. The back of the liner stays put
    because it is what fills the gap under the hair: pulled in, the new head's own neck shows
    through that gap (the space marine, 2026-11-09). Liner vertices with no head behind them --
    a ray that misses -- are dropped with their faces: nothing can hide them.
    """
    moved = vertices.copy()
    empty = np.zeros(len(faces), bool)
    if not liner.any() or radius <= 0:
        return moved, empty

    centre = np.array([axis[0], cut + 0.55 * (head.vertices[:, 1].max() - cut), axis[1]])
    sideways = np.abs(vertices[:, 0] - axis[0])
    behind = np.abs(vertices[:, 2] - axis[1])
    idx = np.flatnonzero(liner & (vertices[:, 1] > cut) & (np.hypot(sideways, behind) < reach * radius)
                         & (sideways > behind))
    if not len(idx):
        return moved, empty

    d = vertices[idx] - centre
    dist = np.linalg.norm(d, axis=1)
    dirs = d / np.maximum(dist, 1e-12)[:, None]
    hits, ray_of, _ = head.ray.intersects_location(
        np.repeat(centre[None], len(idx), 0), dirs, multiple_hits=True)
    # With no hits at all, some trimesh ray engines (Linux CI) return a flat (0,) array, not (0, 3).
    hits = np.asarray(hits, dtype=float).reshape(-1, 3)
    ray_of = np.asarray(ray_of, dtype=int).reshape(-1)
    nearest = np.full(len(idx), np.inf)
    np.minimum.at(nearest, ray_of, np.linalg.norm(hits - centre, axis=1))
    missed = ~np.isfinite(nearest)
    poke = ~missed & (dist > keep * nearest)
    moved[idx[poke]] = centre + dirs[poke] * (keep * nearest[poke])[:, None]

    failed = np.zeros(len(vertices), bool)
    failed[idx[missed]] = True
    return moved, failed[faces].any(axis=1)


def clip_below(mesh, gone: np.ndarray, cut: float):
    """MESH without its GONE vertices, but the triangles that cross from kept to gone are cut along the
    plane y = CUT instead of thrown away whole, so the edge left is a straight line, not a saw-tooth.
    New points (positions and UVs) sit on the crossing edges, shared between the triangles that meet
    there. Only those triangles change."""
    import trimesh

    v, f = mesh.vertices, mesh.faces
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(v)
    g = gone[f]
    whole = f[~g.any(axis=1)]
    cross = f[g.any(axis=1) & ~g.all(axis=1)]
    new_pos, new_uv, made = [], [], {}

    def on_edge(a, b):
        key = (min(a, b), max(a, b))
        if key not in made:
            t = (cut - v[a, 1]) / (v[b, 1] - v[a, 1]) if v[b, 1] != v[a, 1] else 0.5
            t = float(np.clip(t, 0.0, 1.0))
            made[key] = len(v) + len(new_pos)
            new_pos.append(v[a] + t * (v[b] - v[a]))
            if has_uv:
                new_uv.append(uv[a] + t * (uv[b] - uv[a]))
        return made[key]

    tris = []
    for tri in cross:
        poly = []
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            if not gone[a]:
                poly.append(a)
            if gone[a] != gone[b]:
                poly.append(on_edge(a, b))
        tris += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
    faces = np.vstack([whole, np.array(tris, dtype=np.int64).reshape(-1, 3)])
    verts = np.vstack([v, np.array(new_pos).reshape(-1, 3)])
    visual = mesh.visual.copy()
    if has_uv:
        visual.uv = np.vstack([uv, np.array(new_uv).reshape(-1, 2)])
    out = trimesh.Trimesh(verts, faces, visual=visual, process=False)
    out.remove_unreferenced_vertices()
    return out


NECK_BACK_SPAN = 0.35     # neck back match: look this share of the head's height above the cut
NECK_BACK_BAND = 0.3      # points within this share of the neck radius of the neck's centre line (side view)


def back_line(vertices: np.ndarray, x: float, band: float, heights: np.ndarray, step: float) -> np.ndarray:
    """The back of VERTICES (lowest z) on the centre line X (+- BAND) at each of HEIGHTS; NaN where empty."""
    out = np.full(len(heights), np.nan)
    mid = np.abs(vertices[:, 0] - x) < band
    for i, y in enumerate(heights):
        s = mid & (np.abs(vertices[:, 1] - y) < step)
        if s.sum() >= 3:
            out[i] = np.percentile(vertices[s, 2], 2)
    return out


def neck_back_gap(head: np.ndarray, body: np.ndarray, cut: float, axis, ring: float, span: float) -> float:
    """How far the back of HEAD's neck stands in front of BODY's (the old neck, just above CUT, where no collar
    is), on the neck's centre line: the median over the lowest tenth of SPAN. The new head's neck often stands
    more upright, so the old neck shows behind it; sliding the whole head back by this puts it on top."""
    step = 0.01 * span / NECK_BACK_SPAN
    heights = np.arange(cut, cut + 0.1 * span, step)
    band = NECK_BACK_BAND * ring
    gap = back_line(head, axis[0], band, heights, step) - back_line(body, axis[0], band, heights, step)
    gap = gap[~np.isnan(gap)]
    return max(float(np.median(gap)), 0.0) if len(gap) else 0.0


SCRAP_SHARE = 0.3          # a new-head piece this much smaller than the largest, reaching below the cut, is a scrap


def drop_scraps(mesh, cut: float, axis=None, ring: float | None = None):
    """MESH without the loose pieces that reach below CUT and are under SCRAP_SHARE of its largest piece:
    bits of the crop's collar, cut free by the outline. The face and the skull can be separate pieces of
    similar size, so only small ones go. With AXIS (x, z) and RING (radius), a small piece goes only if
    it sticks out beyond RING below the cut. Pieces are what is joined in space: a glTF mesh is split at
    every texture seam, and those patches are not loose."""
    import trimesh

    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    label = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(welded.faces))
    sizes = np.bincount(label)
    if len(sizes) < 2:
        return mesh
    keep = np.ones(len(mesh.faces), bool)
    for piece in np.flatnonzero(sizes < SCRAP_SHARE * sizes.max()):
        faces = np.flatnonzero(label == piece)
        pts = mesh.vertices[np.unique(mesh.faces[faces])]
        low = pts[pts[:, 1] < cut]
        if not len(low):
            continue
        if axis is not None and ring is not None:
            reach = np.hypot(low[:, 0] - axis[0], low[:, 2] - axis[1])
            if float(np.percentile(reach, 90)) <= ring:
                continue
        keep[faces] = False
    return mesh.submesh([np.flatnonzero(keep)], append=True)


BRIDGE_RAYS = 72          # neck bridge: each neck is measured along this many rays from its centre line
BRIDGE_REACH = 1.6        # a neck ring stays within this many neck radii, else the slice is shoulders or a chin
BRIDGE_MISSING = 0.1      # and at most this share of the rays may find nothing (a small hole)
BRIDGE_STEP = 0.005       # the search for each ring moves by this share of the head's height
BRIDGE_GAP = 0.02         # the least height between the two rings, same measure
BRIDGE_SEARCH = 0.25      # how far from the cut either ring may sit, same measure
BRIDGE_TALL = 0.15        # a bridge taller than this is worth a human's glance
BRIDGE_INSET = 0.985      # the bridge sits this far inside both necks, out of sight,
BRIDGE_TUCK = 0.03        # and runs this share of the head's height past each cut edge
BRIDGE_WALL = 0.12        # along a ray from the centre line, a gap wider than this share of the distance so far
                          # ends the neck (its two walls); what lies beyond is a collar
BRIDGE_FLARE = 1.5        # the body's ring is never taken as wider than this many times the head's
BRIDGE_BAND = 0.08        # the centre line comes from the lowest this share (of the head's height) of the new head
BRIDGE_SMOOTH = 5         # rays in the running median and mean that even out each ring


def ring_segments(mesh, height: float, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """MESH's cross-section at HEIGHT: its line segments (n, 2, 2) in (x, z) measured from AXIS, and the
    texture position (UV) of each segment end (n, 2, 2), None for a mesh without UVs."""
    import trimesh

    lines, faces = trimesh.intersections.mesh_plane(mesh, [0.0, 1.0, 0.0], [0.0, height, 0.0], return_faces=True)
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(mesh.vertices)
    if not len(lines):
        return np.zeros((0, 2, 2)), (np.zeros((0, 2, 2)) if has_uv else None)
    lines = np.asarray(lines, float)
    ends = None
    if has_uv:
        corners = mesh.triangles[faces]
        with np.errstate(divide="ignore", invalid="ignore"):     # a sliver triangle has no barycentric frame
            ends = np.stack([np.einsum("nk,nkc->nc", trimesh.triangles.points_to_barycentric(corners, lines[:, k]),
                                       np.asarray(uv, float)[mesh.faces[faces]]) for k in (0, 1)], axis=1)
    return lines[:, :, [0, 2]] - np.asarray(axis, float), ends


def ring_reach(segments: np.ndarray, angles: np.ndarray, ends: np.ndarray | None = None):
    """How far from the centre the neck's skin is along each of ANGLES (0 = front, +z; a quarter turn on = +x),
    from a cross-section's SEGMENTS; NaN where a ray meets nothing. With ENDS (the segments' UVs), also the UV
    at each of those points.

    A ray crosses the neck's inner wall, then its skin (a finished model is a double wall), then perhaps a
    collar's two walls. The skin is the last crossing before the first wide gap: one over BRIDGE_WALL of
    the distance already covered."""
    angles = np.asarray(angles, float)
    out = np.full(len(angles), np.nan)
    where = np.full((len(angles), 2), np.nan)
    if len(segments) and len(angles):
        a, e = segments[:, 0], segments[:, 1] - segments[:, 0]
        d = np.stack([np.sin(angles), np.cos(angles)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
            t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
            s = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den
        t[~((t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9))] = np.inf      # misses and parallel segments
        order = np.argsort(t, axis=1)
        near = np.take_along_axis(t, order, axis=1)
        far = np.append(near[:, 1:], np.full((len(angles), 1), np.inf), axis=1)
        with np.errstate(invalid="ignore"):
            last = np.argmax(far - near > BRIDGE_WALL * near, axis=1)     # the first wide gap (or the end)
        rows = np.arange(len(angles))
        best = order[rows, last]
        hit = np.isfinite(near[rows, last])
        out[hit] = near[rows, last][hit]
        if ends is not None:
            along = np.clip(s[rows, best], 0.0, 1.0)[:, None]
            where[hit] = (ends[best, 0] + along * (ends[best, 1] - ends[best, 0]))[hit]
    return out if ends is None else (out, where)


def bridge_angles() -> np.ndarray:
    """The angles a neck ring is measured at (see `ring_reach`)."""
    return np.linspace(-np.pi, np.pi, BRIDGE_RAYS, endpoint=False)


def bridge_ring(mesh, height: float, axis: np.ndarray, reach: float,
              missing: float = BRIDGE_MISSING) -> np.ndarray | None:
    """MESH's neck at HEIGHT as a radius from AXIS at each of `bridge_angles`, evened out; None if the slice
    is not a neck: over MISSING of the rays find nothing (a ragged edge), or it runs further than REACH
    (the shoulders, a chin). A few missing rays take their neighbours' radius; a running median then drops
    the odd ray that caught a collar or the inner wall."""
    angles = bridge_angles()
    radius = ring_reach(ring_segments(mesh, height, axis)[0], angles)
    seen = ~np.isnan(radius)
    if seen.mean() < 1 - missing or radius[seen].max() > reach:
        return None
    radius = np.interp(angles, angles[seen], radius[seen], period=2 * np.pi)
    radius = ndimage.median_filter(radius, size=BRIDGE_SMOOTH, mode="wrap")
    return ndimage.uniform_filter1d(radius, size=BRIDGE_SMOOTH, mode="wrap")


def ring_angle(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The angle of each of POINTS (x, y, z) round AXIS (x, z), as `ring_reach` counts them."""
    return np.arctan2(points[:, 0] - axis[0], points[:, 2] - axis[1])


def cut_neck(mesh, height: float, axis: np.ndarray, limit, above: bool = True):
    """MESH without its neck on one side of HEIGHT (above it, or below), cut straight: what stands there no
    further from AXIS than LIMIT (one radius, or one for each of `bridge_angles`). What is further out (a
    collar, the shoulders) stays; the few triangles that join the two on that side go whole."""
    v = mesh.vertices
    side = v[:, 1] > height if above else v[:, 1] < height
    if np.ndim(limit):
        limit = np.interp(ring_angle(v, axis), bridge_angles(), limit, period=2 * np.pi)
    gone = side & (np.hypot(v[:, 0] - axis[0], v[:, 2] - axis[1]) < limit)
    f = mesh.faces
    trimmed = mesh.copy()
    trimmed.update_faces(~(gone[f].any(axis=1) & (side & ~gone)[f].any(axis=1)))
    return clip_below(trimmed, gone, height)


def neck_centre(head, head_height: float) -> np.ndarray:
    """The centre line (x, z) of a grafted head's neck: the middle of its lowest BRIDGE_BAND. The graft's own
    axis (`neck_core`) leans towards the chin, which is fine for splitting neck from collar by distance but
    not for looking round the neck from inside it."""
    v = head.vertices
    band = v[v[:, 1] < v[:, 1].min() + BRIDGE_BAND * head_height][:, [0, 2]]
    return (np.percentile(band, 2, axis=0) + np.percentile(band, 98, axis=0)) / 2


def neck_paint(head, height: float, axis: np.ndarray) -> np.ndarray | None:
    """One texture position (UV) for the whole bridge: HEAD's own neck skin at HEIGHT, at the point whose
    colour is closest to the usual one (the median) across the front half. One colour, because the ring
    runs through the shadow under the jaw and through scuffed paint, and a bridge painted angle by angle
    came out striped. The front half, because the back of a head built from a front crop is often pale.
    None for a head without UVs."""
    segments, ends = ring_segments(head, height, axis)
    if ends is None:
        return None
    angles = np.linspace(-np.pi / 2, np.pi / 2, 2 * BRIDGE_RAYS)
    radius, uv = ring_reach(segments, angles, ends)
    uv = uv[~np.isnan(radius) & ~np.isnan(uv).any(axis=1)]
    if not len(uv):
        return None
    image = getattr(head.visual.material, "baseColorTexture", None)
    if image is None:
        return uv[len(uv) // 2]
    colour = sample_texture(image, uv)
    return uv[np.linalg.norm(colour - np.median(colour, axis=0), axis=1).argmin()]


def bridge_neck(body, head, cut: float, ring: float, head_height: float):
    """Close the seam between BODY's neck and the new HEAD's: (body, head, report).

    After the graft the two necks overlap as two ragged edges that are not joined, and an open neckline
    shows holes and the old stump between them. Both are cut straight where each is still a whole ring
    (the body's just under CUT, the head's the lowest it has), and a smooth tube of triangles is added to
    the head between the two. The tube sits a little inside both necks and runs on past each cut edge (in
    the head, up to where its neck has no holes), so no edge has to meet another and what shows between them is the tube. It has one colour, the
    head's own neck skin: the body's paint under the old chin is shadow. A collar round the neck is left
    alone (`ring_reach`). RING is the neck's radius; HEAD_HEIGHT sets the scale. Nothing changes, and the
    report says why, when either neck has no whole ring near the cut, or when the head's neck already
    reaches down into the body all the way round (no seam shows)."""
    import trimesh

    axis = neck_centre(head, head_height)
    reach, step = BRIDGE_REACH * ring, BRIDGE_STEP * head_height
    span, gap, tuck = BRIDGE_SEARCH * head_height, BRIDGE_GAP * head_height, BRIDGE_TUCK * head_height

    def find(mesh, heights, missing=BRIDGE_MISSING):
        for y in heights:
            radius = bridge_ring(mesh, float(y), axis, reach, missing)
            if radius is not None:
                return float(y), radius
        return None, None

    low, foot = find(body, np.arange(cut - step, cut - span, -step))
    if low is None:
        return body, head, {"skipped": "the body's neck is not a whole ring under the cut"}
    if bridge_ring(head, low, axis, reach) is not None:
        # e.g. a head that sits down inside a collar: the seam is out of sight and a bridge would only add edges
        return body, head, {"skipped": "the new head's neck already reaches into the body all the way round",
                            "expected": True}
    start = max(low + gap, float(head.vertices[:, 1].min()) + step)
    high, top = find(head, np.arange(start, cut + span, step))
    if high is None:
        return body, head, {"skipped": "the new head has no whole ring of neck under its chin"}
    # The head's ring may have small holes, and its ragged edge can reach a little higher still: the tube
    # runs on inside the head to where its neck has none.
    sealed, inner = find(head, np.arange(high + step, cut + span, step), missing=0.0)
    if sealed is None:
        sealed, inner = high, top
    foot = np.minimum(foot, BRIDGE_FLARE * top)
    spot = neck_paint(head, high, axis)
    body = cut_neck(body, low, axis, (1 + BRIDGE_WALL / 2) * foot)
    head = cut_neck(head, high, axis, reach, above=False)

    angles = bridge_angles()
    direction = np.stack([np.sin(angles), np.cos(angles)], axis=1)
    rows = [(low - tuck, foot), (low, foot), (high, top), (sealed + tuck, inner)]
    points = np.concatenate([np.column_stack([axis[0] + BRIDGE_INSET * r * direction[:, 0], np.full(len(angles), y),
                                              axis[1] + BRIDGE_INSET * r * direction[:, 1]]) for y, r in rows])
    n, first = len(angles), len(head.vertices)
    here = np.arange(n)
    there = (here + 1) % n
    tris = np.concatenate([np.concatenate([
        np.column_stack([first + (k + 1) * n + here, first + k * n + here, first + k * n + there]),
        np.column_stack([first + (k + 1) * n + here, first + k * n + there, first + (k + 1) * n + there]),
    ]) for k in range(len(rows) - 1)])
    visual = head.visual.copy()
    if spot is not None:
        visual.uv = np.vstack([head.visual.uv, np.tile(spot, (len(points), 1))])
    head = trimesh.Trimesh(np.vstack([head.vertices, points]), np.vstack([head.faces, tris]),
                           visual=visual, process=False)
    return body, head, {"body_at": round(low, 4), "head_at": round(high, 4), "faces": int(len(tris)),
                        "centre": [round(float(c), 4) for c in axis],
                        "height_share": round((high - low) / head_height, 3)}


def bridge_notes(bridge: dict | None) -> list[str]:
    """What a person should check about a neck bridge (`bridge_neck`'s report); empty when it went well."""
    if not bridge:
        return []
    if "skipped" in bridge:
        return [] if bridge.get("expected") else [f"the neck seam was not bridged: {bridge['skipped']}"]
    if bridge["height_share"] > BRIDGE_TALL:
        return [f"the neck bridge is tall ({bridge['height_share']:.0%} of the head's height): check the neck"]
    return []


def graft(full, head, full_box, head_box, crop, head_width, bone, shape, grow=None, liner=LINER_SHRINK,
          cut_at=None, skull_fit=False, neck_bridge=False):
    """Swap FULL's head region for HEAD (both trimesh meshes; HEAD is moved in place).

    CUT_AT: "neck" finds the neck (`neck_height`) and swaps only above it; a height (model
    units) cuts there, e.g. one a person marked; None swaps the whole head region.
    NECK_BRIDGE closes the seam between the two necks (`bridge_neck`) when the cut sits under the
    jaw (a bare neck or a tight collar); a wide collar hides its seam and is left alone.

    1. The real head: FULL's head bones. 2. HEAD placed by the crop box, then on the real
    head at the same centre and height. 3. Fine line-up, position only. 4. FULL loses the
    front half of its head region (the back half stays, shrunk by LINER, as a liner under
    the new hair, tucked inside HEAD; LINER=0 drops it all); HEAD keeps what lands inside the
    real head's outline, grown a little. Returns (body, head part, report).
    """
    import trimesh

    picked = head_bones(bone, full.vertices[:, 1])
    real = np.isin(bone, picked)
    full_frame = frame(full.vertices, full_box)
    fpx, fpy = project(full.vertices, full_frame)
    outline = silhouette(full.faces[real[full.faces].all(axis=1)], fpx, fpy, shape)

    head.apply_transform(crop_to_full(full_frame, frame(head.vertices, head_box), crop, head_width))
    new = on_mask(*project(head.vertices, full_frame), outline)
    if not new.any():
        raise ValueError("the head model does not land on the body's head: check the crop")
    place = place_on(full.vertices[real], head.vertices[new])
    head.apply_transform(place)

    rng = np.random.default_rng(0)
    new = on_mask(*project(head.vertices, full_frame), outline)
    src, dst = head.vertices[new], full.vertices[real]
    src = src[rng.choice(len(src), min(30000, len(src)), replace=False)]
    dst = dst[rng.choice(len(dst), min(80000, len(dst)), replace=False)]
    fine, _, cost = trimesh.registration.icp(src, dst, max_iterations=60, scale=False)
    head.apply_transform(fine)
    if skull_fit:
        # The crop sizes the new head only roughly, and the height match above includes the neck, so a
        # head a few percent too big sits low and forward at the chin. Line up the skulls (upper part of
        # both heads, where they agree) with scale; the neck and chin then follow.
        top_r, top_n = full.vertices[real, 1].max(), head.vertices[new, 1].max()
        hr, hn = np.ptp(full.vertices[real, 1]), np.ptp(head.vertices[new, 1])
        s_src = head.vertices[new & (head.vertices[:, 1] > top_n - SKULL_SHARE * hn)]
        s_dst = full.vertices[real & (full.vertices[:, 1] > top_r - SKULL_SHARE * hr)]
        s_src = s_src[rng.choice(len(s_src), min(30000, len(s_src)), replace=False)]
        s_dst = s_dst[rng.choice(len(s_dst), min(80000, len(s_dst)), replace=False)]
        skull, _, cost = trimesh.registration.icp(s_src, s_dst, max_iterations=80, scale=True)
        head.apply_transform(skull)
        place = place @ np.diag([np.cbrt(abs(np.linalg.det(skull[:3, :3])))] * 3 + [1.0])

    if grow is None:
        # A margin in proportion to the head: the new head's hair often stands wider than the
        # body's own head, and a few fixed pixels trimmed it into holes at the sides.
        cols = np.flatnonzero(outline.any(axis=0))
        grow = max(OUTLINE_GROW_PX, int(round(OUTLINE_GROW_SHARE * (cols.max() - cols.min()))))
    keep = on_mask(*project(head.vertices, full_frame), ndimage.binary_dilation(outline, iterations=grow))
    tight = False
    if cut_at == "neck":
        cut = neck_height(full.vertices[real, 1], full.vertices[real, 0])
        if cut is None:   # no collar inside the head region: look at the body's own profile below it
            tight = True
            cut = collar_top(full.vertices, real)
            chin = chin_bottom(full.vertices, real)
            if chin is not None:
                cut = min(cut, chin - CHIN_CLEAR * float(np.ptp(full.vertices[real, 1])))
    else:
        cut = cut_at
    if cut is None:
        axis, ring = None, None
    else:
        axis, ring = neck_core(full.vertices, real) if tight else neck_ring(full.vertices, real, cut)
    swap, in_head = head_cut(full.vertices, real, head.vertices, cut, axis, ring)
    if tight:
        # The cut sits under the jaw, so the plane alone takes the whole old head; below it the body keeps
        # its neck and collar as they are (no radius split: a tight collar is barely wider than the neck).
        swap = real & (full.vertices[:, 1] >= cut)
    keep &= in_head
    body_mesh = full.copy()
    body_mesh.vertices, gone = liner_split(full.vertices, swap, liner)
    # The liner is only safe if it stays inside the new head: an old head with more volume pokes
    # out beside the new ears. Above the neck and near the skull only, or the collar gets pulled
    # up with it (2026-10-30). With no neck cut, start the guard at the head region's floor.
    above = cut if cut is not None else float(np.percentile(full.vertices[real, 1], 2))
    if axis is None:
        axis, ring = neck_ring(full.vertices, real, above)
    before = body_mesh.vertices.copy()
    body_mesh.vertices, liner_miss = tuck_liner(before, full.faces, swap & ~gone, head, above,
                                                axis, ring)
    tucked = int(np.any(body_mesh.vertices != before, axis=1).sum())
    if tight and not liner_miss.any():   # a straight edge where the old neck ends, not a saw-tooth of whole triangles
        body = clip_below(body_mesh, gone, cut)
    else:
        drop = gone[full.faces].any(axis=1) | liner_miss
        body = body_mesh.submesh([np.flatnonzero(~drop)], append=True)
    part = head.submesh([np.flatnonzero(keep[head.faces].all(axis=1))], append=True)
    pushed = 0.0
    if tight:
        part = drop_scraps(part, cut, axis, ring)
        # After the scraps are picked (they are judged by distance from the neck, which this changes). The
        # whole body, not only the head bone's share: that starts above the cut, on the skull. One straight
        # slide back: the head keeps its shape.
        pushed = neck_back_gap(part.vertices, full.vertices, cut, axis, ring,
                               NECK_BACK_SPAN * float(np.ptp(full.vertices[real, 1])))
        part.vertices = part.vertices - [0.0, 0.0, pushed]
    bridge = None
    if neck_bridge:
        if tight:
            body, part, bridge = bridge_neck(body, part, cut, ring, float(np.ptp(full.vertices[real, 1])))
        else:
            bridge = {"skipped": "the cut is not under the jaw (a wide collar, a marked height or the whole head)",
                      "expected": True}
    report = {"head_bones": picked, "outline_margin_px": int(grow), "liner_shrink": liner,
              "liner_tucked": tucked, "liner_dropped_faces": int(liner_miss.sum()),
              "cut_at": None if cut is None else round(cut, 4),
              "neck_ring": None if ring is None else round(ring, 4),
              "slid_back": round(pushed, 4), "neck_bridge": bridge,
              "scale_to_real_head": round(float(place[0, 0]), 4),
              "line_up_cost": float(cost), "body_faces": int(len(body.faces)),
              "head_faces": int(len(part.faces))}
    return body, part, report


MATERIAL_TOLERANCE = 0.3   # a change this big is worth a human's glance
SPREAD_LIMITS = (0.8, 1.25)  # how far matching may stretch or squeeze a channel's contrast


def sample_texture(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    """Texture values (0..1, RGB) at glTF UVs (v runs up the image)."""
    a = np.asarray(image.convert("RGB")).astype(float) / 255
    h, w = a.shape[:2]
    u, v = np.clip(uv[:, 0], 0, 1), np.clip(uv[:, 1], 0, 1)   # atlases sit in 0..1; 1.0 must not wrap to 0
    x = np.round(u * (w - 1)).astype(int)
    y = np.round((1 - v) * (h - 1)).astype(int)
    return a[y, x]


def paired_points(head_vertices: np.ndarray, body_vertices: np.ndarray, reach: float):
    """(head index, body index) pairs of points that sit on the same spot, within REACH.

    The two heads are lined up, so the nearest body point is the same part of the character:
    hair against hair, skin against skin. Comparing whole regions instead mixed things up
    (the mage's body head region took in some of her bright hood, and her black hair was
    'corrected' to brown and her face to white).
    """
    distance, nearest = cKDTree(body_vertices).query(head_vertices, workers=-1)
    close = np.flatnonzero(distance < reach)
    return close, nearest[close]


def match_to_body(head_texture: Image.Image, head_seen: np.ndarray, body_seen: np.ndarray,
                  channels: tuple[int, ...] = (0, 1, 2)) -> tuple[Image.Image, dict]:
    """HEAD_TEXTURE with each channel's level and contrast moved to the body's.

    The head and body are built in separate runs and come out in different states: one
    woman's head crop was called all metal (1.0) while her body said cloth (0.0), and its
    paint sat darker and more saturated. A hero must look like one piece, so the head always
    follows the body, which is most of the character. HEAD_SEEN and BODY_SEEN are the two
    textures sampled at paired points on the same head (0..1 RGB rows, row i of each is the
    same spot); a channel with no spread in the head (all metal) simply lands on the body's
    level. Contrast changes are kept within SPREAD_LIMITS.
    """
    a = np.asarray(head_texture.convert("RGB")).astype(float) / 255
    report = {}
    for c in channels:
        mh, sh = float(head_seen[:, c].mean()), float(head_seen[:, c].std())
        mb = float(body_seen[:, c].mean())
        if sh > 1e-3:
            # A straight-line fit over the paired points (body = stretch * head + shift):
            # matching spreads instead let a few bright points blow light skin out to white.
            slope = float(np.polyfit(head_seen[:, c], body_seen[:, c], 1)[0])
            stretch = float(np.clip(slope, *SPREAD_LIMITS))
        else:
            stretch = 0.0
        a[..., c] = (a[..., c] - mh) * stretch + mb
        report["rgb"[c]] = {"head": round(mh, 3), "body": round(mb, 3), "stretch": round(stretch, 2)}
    out = Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")
    report["big_change"] = any(abs(v["head"] - v["body"]) > MATERIAL_TOLERANCE for v in report.values())
    return out, report


def load_picture(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGBA") if opened.mode in ("RGBA", "LA", "P") else opened.convert("RGB")
