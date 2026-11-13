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


def clip_below(mesh, gone: np.ndarray, cut: float, above_whole: bool = False):
    """MESH without its GONE vertices, but the triangles that cross from kept to gone are cut along the
    plane y = CUT instead of thrown away whole, so the edge left is a straight line, not a saw-tooth.
    New points (positions and UVs) sit on the crossing edges, shared between the triangles that meet
    there. Only those triangles change. ABOVE_WHOLE: a crossing triangle with no corner under the plane
    (kept collar meeting old skin above the cut) has nothing to be cut along and goes whole."""
    import trimesh

    v, f = mesh.vertices, mesh.faces
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(v)
    g = gone[f]
    whole = f[~g.any(axis=1)]
    cross = f[g.any(axis=1) & ~g.all(axis=1)]
    if above_whole:
        cross = cross[(v[cross, 1] < cut).any(axis=1)]
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


NECK_WIDE = 1.0           # a band of the new head is its neck if it is at least this many neck radii across, both ways


def neck_centre(head, head_height: float, ring: float | None = None) -> np.ndarray:
    """The centre line (x, z) of a grafted head's neck: the middle of its lowest BRIDGE_BAND. The graft's own
    axis (`neck_core`) leans towards the chin, which is fine for splitting neck from collar by distance but
    not for looking round the neck from inside it."""
    v = head.vertices
    floor = float(v[:, 1].min())
    band = v[v[:, 1] < floor + BRIDGE_BAND * head_height][:, [0, 2]]
    if ring is not None:
        # The lowest thing on a head is not always its neck: pc_f01e's ended in a stalk a third of the neck wide,
        # 2 cm to one side, and both the fit and the bridge then measured the necks from their edge (the fit
        # refused, the bridge cut the neck away and left a rod). Go up to the first band as wide as a neck.
        for lift in np.arange(0.0, BRIDGE_SEARCH, BRIDGE_STEP) * head_height:
            higher = v[(v[:, 1] >= floor + lift) & (v[:, 1] < floor + lift + BRIDGE_BAND * head_height)][:, [0, 2]]
            if len(higher) >= MIN_SLICE_POINTS and \
                    (np.percentile(higher, 98, axis=0) - np.percentile(higher, 2, axis=0)).min() >= NECK_WIDE * ring:
                band = higher
                break
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

    axis = neck_centre(head, head_height, ring)
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


FIT_RAYS = 180            # neck fit: the rings are measured along this many rays
FIT_REACH = 0.12          # the new neck changes shape up to this share of the head's height above its cut edge,
FIT_CHIN = 1.25           # but along each ray only up to where something stands this far outside the neck (the jaw,
FIT_CLEAR = 0.015         # an ear, hair), less this share of the head's height
FIT_LEAST = 0.02          # the least height the change is spread over, same measure
FIT_SHARE = 0.5           # with fewer than this share of the rays having that much room, nothing is fitted
FIT_MOST = 0.6            # a ring that would grow or shrink by more than this share was not measured on a neck
FIT_SPREAD = 9            # rays in the running minimum and mean that even out where the change ends
FIT_TUCK = 0.015          # the head's edge ends this share of the head's height down inside the body's neck
FIT_COLLAR = 2.2          # where the body's edge is this much wider than the new neck it is a collar, not neck,
FIT_CLOTH = 0.2           # and so it is where its paint is this far (0..1, any channel) from the new neck's skin
FIT_HUE = 0.06            # and either this far from it in hue too (each channel's share of the colour) or
FIT_LOOSE = 1.2           # this much wider than the new neck: lit against shaded skin differs in brightness only
FIT_COVERED = 0.7         # with more than this share of the ring under a collar there is nothing to aim at
FIT_SHADE = 0.08          # the most (0..1, any channel) the head's neck paint changes towards the body's
FIT_PAINT_SPREAD = 9      # rays in the running mean of the colour difference between the two necks
FIT_PAINT_GROW = 2        # the repainted patch grows this many pixels past its triangles, over the gaps between them
FIT_BODY = 1.0            # the share of the difference the body's neck stump takes first (0: only the head's neck moves)
FIT_DEPTH = 0.25          # the base of the body's neck changes shape down to this share of the head's height under
FIT_WIDE = (1.3, 2.2)     # its edge, in full within the first of these neck radii of the centre line, not at all past the second
FIT_STUMP = 0.01          # a ray with less free skin than this share of the head's height under the edge is left alone
FIT_UV_NEAR = 0.03        # new skin between two rays is painted across them only if their paint sits this close (UV)


def fit_angles() -> np.ndarray:
    """The angles the neck fit measures a ring at (see `ring_reach`)."""
    return np.linspace(-np.pi, np.pi, FIT_RAYS, endpoint=False)


def ring_far(segments: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """The farthest thing a cross-section's SEGMENTS put along each of ANGLES (as `ring_reach` counts them),
    whatever it is: neck, jaw, collar or hair. NaN where a ray meets nothing."""
    angles = np.asarray(angles, float)
    out = np.full(len(angles), np.nan)
    if len(segments) and len(angles):
        a, e = segments[:, 0], segments[:, 1] - segments[:, 0]
        d = np.stack([np.sin(angles), np.cos(angles)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
            t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
            s = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den
        t[~((t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9))] = -np.inf
        far = t.max(axis=1)
        out[np.isfinite(far)] = far[np.isfinite(far)]
    return out


def fit_ring(mesh, height: float, axis: np.ndarray, reach: float,
             missing: float = BRIDGE_MISSING) -> np.ndarray | None:
    """MESH's neck skin at HEIGHT as a radius from AXIS at each of `fit_angles`, as it is (only the odd ray
    that caught something else is dropped); None if the slice is not a whole neck (see `bridge_ring`)."""
    angles = fit_angles()
    radius = ring_reach(ring_segments(mesh, height, axis)[0], angles)
    seen = ~np.isnan(radius)
    if seen.mean() < 1 - missing or radius[seen].max() > reach:
        return None
    radius = np.interp(angles, angles[seen], radius[seen], period=2 * np.pi)
    return ndimage.median_filter(radius, size=BRIDGE_SMOOTH, mode="wrap")


def split_at(mesh, height: float, axis=None, within: float | None = None):
    """MESH with every triangle that crosses the plane y = HEIGHT cut in two along it, so a ring of points
    sits on the plane. New points (positions and UVs) are shared between the triangles that meet there; the
    shape does not change. With AXIS (x, z) and WITHIN, only triangles with a corner that close to the axis."""
    import trimesh

    v, f = mesh.vertices, mesh.faces
    uv = getattr(mesh.visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(v)
    up = v[:, 1] >= height
    crossing = up[f].any(axis=1) & ~up[f].all(axis=1)
    if axis is not None and within is not None:
        crossing &= (np.hypot(v[:, 0] - axis[0], v[:, 2] - axis[1]) < within)[f].any(axis=1)
    if not crossing.any():
        return mesh
    new_pos, new_uv, made = [], [], {}

    def on_edge(a, b):
        key = (min(a, b), max(a, b))
        if key not in made:
            t = float(np.clip((height - v[a, 1]) / (v[b, 1] - v[a, 1]), 0.0, 1.0))
            made[key] = len(v) + len(new_pos)
            new_pos.append(v[a] + t * (v[b] - v[a]))
            if has_uv:
                new_uv.append(uv[a] + t * (uv[b] - uv[a]))
        return made[key]

    tris = []
    for tri in f[crossing]:
        sides = ([], [])
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            sides[int(up[a])].append(a)
            if up[a] != up[b]:
                point = on_edge(a, b)
                sides[0].append(point)
                sides[1].append(point)
        for poly in sides:
            tris += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
    visual = mesh.visual.copy()
    if has_uv:
        visual.uv = np.vstack([uv, np.array(new_uv).reshape(-1, 2)])
    return trimesh.Trimesh(np.vstack([v, np.array(new_pos).reshape(-1, 3)]),
                           np.vstack([f[~crossing], np.array(tris, dtype=np.int64).reshape(-1, 3)]),
                           visual=visual, process=False)


def ring_paint(mesh, heights, axis: np.ndarray, angles: np.ndarray) -> np.ndarray | None:
    """The colour (0..1 RGB) of MESH's neck skin along each of ANGLES, the median over HEIGHTS; NaN where a
    ray meets nothing. None for a mesh without a painted picture."""
    import warnings

    image = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    rows = []
    for y in heights:
        segments, ends = ring_segments(mesh, float(y), axis)
        if image is None or ends is None:
            return None
        radius, uv = ring_reach(segments, angles, ends)
        seen = ~np.isnan(radius) & ~np.isnan(uv).any(axis=1)
        row = np.full((len(angles), 3), np.nan)
        row[seen] = sample_texture(image, uv[seen])
        rows.append(row)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a ray that is empty at every height
        return np.nanmedian(np.stack(rows), axis=0)


def tint_paint(mesh, tint: np.ndarray):
    """MESH's painted picture with TINT added: a colour change (RGB, 0..1) for each point, spread smoothly
    across every triangle that has one. Returns the new picture, or None for a mesh without one. The patch
    grows a little past its triangles so the unpainted gaps between texture islands do not show as lines."""
    image = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)
    if image is None or uv is None or len(uv) != len(mesh.vertices):
        return None
    paint = np.asarray(image.convert("RGB")).astype(float) / 255
    h, w = paint.shape[:2]
    px, py = np.clip(uv[:, 0], 0, 1) * (w - 1), (1 - np.clip(uv[:, 1], 0, 1)) * (h - 1)
    change = np.zeros_like(paint)
    done = np.zeros((h, w), bool)
    for a, b, c in mesh.faces[np.abs(tint[mesh.faces]).sum(axis=(1, 2)) > 0]:
        x0, x1 = int(np.floor(min(px[a], px[b], px[c]))), int(np.ceil(max(px[a], px[b], px[c])))
        y0, y1 = int(np.floor(min(py[a], py[b], py[c]))), int(np.ceil(max(py[a], py[b], py[c])))
        den = (py[b] - py[c]) * (px[a] - px[c]) + (px[c] - px[b]) * (py[a] - py[c])
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=float), np.arange(y0, y1 + 1, dtype=float))
        wa = ((py[b] - py[c]) * (gx - px[c]) + (px[c] - px[b]) * (gy - py[c])) / den
        wb = ((py[c] - py[a]) * (gx - px[c]) + (px[a] - px[c]) * (gy - py[c])) / den
        wc = 1 - wa - wb
        inside = (wa >= -0.02) & (wb >= -0.02) & (wc >= -0.02)
        value = wa[..., None] * tint[a] + wb[..., None] * tint[b] + wc[..., None] * tint[c]
        change[y0:y1 + 1, x0:x1 + 1][inside] = value[inside]
        done[y0:y1 + 1, x0:x1 + 1][inside] = True
    if not done.any():
        return image
    far, nearest = ndimage.distance_transform_edt(~done, return_indices=True)
    grown = (far > 0) & (far <= FIT_PAINT_GROW)
    change[grown] = change[nearest[0][grown], nearest[1][grown]]
    out = Image.fromarray(np.round(np.clip(paint + change, 0, 1) * 255).astype(np.uint8), "RGB")
    if image.mode == "RGBA":
        out.putalpha(image.getchannel("A"))
    return out


def stump_room(body, low: float, foot: np.ndarray, axis: np.ndarray, head_height: float) -> np.ndarray:
    """How much free neck skin BODY has under its edge at LOW along each of `fit_angles`: the height down to the
    first slice where something stands FIT_CHIN outside the edge's ring FOOT (a collar, the shoulders), where
    the skin is missing, or where the paint is no longer the edge's own colour (cloth), less FIT_CLEAR; at
    most FIT_REACH of HEAD_HEIGHT. Evened out round the neck towards the shortest."""
    angles = fit_angles()
    step, limit = BRIDGE_STEP * head_height, FIT_REACH * head_height
    image = getattr(getattr(body.visual, "material", None), "baseColorTexture", None)
    bottom = np.full(len(angles), low - limit)
    rim = None
    for y in np.arange(low - step, low - limit, -step):
        segments, ends = ring_segments(body, float(y), axis)
        far = ring_far(segments, angles)
        with np.errstate(invalid="ignore"):
            stop = np.isnan(far) | (far > FIT_CHIN * foot)
        if image is not None and ends is not None:
            radius, uv = ring_reach(segments, angles, ends)
            seen = ~np.isnan(radius) & ~np.isnan(uv).any(axis=1)
            colour = np.full((len(angles), 3), np.nan)
            colour[seen] = sample_texture(image, uv[seen])
            rim = colour if rim is None else np.where(np.isnan(rim), colour, rim)
            with np.errstate(invalid="ignore"):
                stop |= np.abs(colour - rim).max(axis=1) > FIT_CLOTH
        bottom = np.where(stop & (bottom < y), y, bottom)
    room = np.clip(low - bottom - FIT_CLEAR * head_height, 0.0, None)
    return ndimage.uniform_filter1d(ndimage.minimum_filter1d(room, FIT_SPREAD, mode="wrap"), FIT_SPREAD, mode="wrap")


def fit_stump(body, low: float, foot: np.ndarray, target: np.ndarray, rise: float, axis: np.ndarray,
              head_height: float, shade: np.ndarray | None = None, depth: float | None = None,
              collared: bool = False):
    """Bring the base of BODY's neck to the ring TARGET: (body, report).

    BODY ends in a straight edge at LOW (ring FOOT). The whole base of the neck under it -- skin, collar, both
    walls -- shrinks towards (or grows from) the neck's centre line as one smooth field: each point's distance
    from the centre line is scaled by TARGET / FOOT for its angle, in full at the edge and near the neck, and
    less and less with depth (to nothing DEPTH of HEAD_HEIGHT under the edge) and with distance from the
    centre line (to nothing FIT_WIDE[1] neck radii out), so the shoulders and chest stay as they are. COLLARED
    says part of the edge is under a collar: the field then ends sooner wherever a collar stands clear of the
    neck at the edge, so an open collar keeps its shape while the neck inside it changes. One
    field for everything tears nothing and keeps the two walls together; a collar ends up hugging the new
    neck. Paint moves with the points. With RISE the neck then grows up by that much: a new strip of skin from
    the edge to TARGET, each part of it painted like the edge's skin right under it. SHADE (a colour change
    for each of `fit_angles`) is added to the paint of the free skin under the edge (`stump_room`), in full at
    the edge and fading out below. The report has the ring reached at the old edge (`rim`), the new edge's
    height (`edge`), how far down anything changed (`lowest`) and the largest move of a point that is not
    free neck skin (`cloth_move`)."""
    import trimesh

    angles = fit_angles()
    step, tuck = BRIDGE_STEP * head_height, FIT_TUCK * head_height
    deep = (FIT_DEPTH if depth is None else depth) * head_height
    room = stump_room(body, low, foot, axis, head_height)
    usable = ndimage.uniform_filter1d((room >= FIT_STUMP * head_height).astype(float), FIT_SPREAD, mode="wrap")
    factor = np.ones(len(angles))
    if deep > 0:                                           # (the difference is evened out, so no change stays exactly none)
        factor = 1 + ndimage.uniform_filter1d(target / foot - 1, FIT_SPREAD, mode="wrap")
    rim = foot * factor

    v = np.array(body.vertices, float)
    at = ring_angle(v, axis)

    def along(values):
        return np.interp(at, angles, values, period=2 * np.pi)

    def ease(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    off = v[:, [0, 2]] - axis
    radius = np.hypot(off[:, 0], off[:, 1])
    v_foot = along(foot)
    down = ease((v[:, 1] - (low - deep)) / deep) if deep > 0 else np.zeros(len(v))
    # Where something stands clear of the neck at the edge (a collar with air behind it), the field ends before it.
    far = ring_far(ring_segments(body, low - 0.2 * step, axis)[0], angles)
    with np.errstate(invalid="ignore"):
        apart = (far > FIT_CHIN * foot) & collared
    wide = np.full(len(angles), FIT_WIDE[1])
    wide[apart] = np.minimum(FIT_WIDE[1], 0.95 * far[apart] / foot[apart])
    if apart.any():
        wide = ndimage.uniform_filter1d(ndimage.minimum_filter1d(wide, FIT_SPREAD, mode="wrap"), FIT_SPREAD, mode="wrap")
    v_wide = along(wide)
    v_near = np.minimum(FIT_WIDE[0], 0.5 * (1 + v_wide))
    out = 1 - ease((radius / v_foot - v_near) / (v_wide - v_near))
    scale = 1 + down * out * (along(factor) - 1)
    changed = scale != 1                                  # every other point keeps its exact place
    v[changed] = np.column_stack([axis[0] + off[changed, 0] * scale[changed], v[changed, 1],
                                  axis[1] + off[changed, 1] * scale[changed]])
    shift = radius * np.abs(scale - 1)
    v_room = np.maximum(along(room), 1e-9)
    skin = (v[:, 1] <= low + 1e-9) & (radius <= FIT_CHIN * v_foot) & (v[:, 1] >= low - v_room)
    weight = np.where(skin, ease((v[:, 1] - (low - v_room)) / v_room), 0.0) * along(usable)
    touched = shift > 1e-9
    cloth = float(shift[~skin].max()) if (~skin).any() else 0.0
    visual = body.visual.copy()
    painted = False
    if shade is not None:
        tint = weight[:, None] * np.stack([along(shade[:, k]) for k in range(3)], axis=1)
        picture = tint_paint(body, tint)
        if picture is not None:
            material = body.visual.material.copy()
            material.baseColorTexture = picture
            visual = trimesh.visual.TextureVisuals(uv=body.visual.uv, material=material)
            painted = True
            touched |= np.abs(tint).sum(axis=1) > 0
    faces, added = body.faces, 0
    if rise > 0:
        # The new strip: one row tucked just inside the stump, one on its edge, one on the target ring above.
        rays = bridge_angles()
        n = len(rays)
        shaped = trimesh.Trimesh(v, body.faces, visual=body.visual, process=False)
        segments, ends = ring_segments(shaped, low - 0.5 * step, axis)
        under = ring_reach(ring_segments(shaped, low - tuck, axis)[0], rays)
        r_rim = np.interp(rays, angles, rim, period=2 * np.pi)
        rows = [(low - tuck, BRIDGE_INSET * np.fmin(r_rim, under)), (low, r_rim),
                (low + rise, np.interp(rays, angles, target, period=2 * np.pi))]
        direction = np.stack([np.sin(rays), np.cos(rays)], axis=1)
        ring_points = [np.column_stack([axis[0] + r * direction[:, 0], np.full(n, y), axis[1] + r * direction[:, 1]])
                       for y, r in rows]
        uv = None
        if ends is not None:
            skin, uv = ring_reach(segments, rays, ends)
            seen = np.flatnonzero(~np.isnan(skin) & ~np.isnan(uv).any(axis=1))
            image = getattr(getattr(body.visual, "material", None), "baseColorTexture", None)
            if len(seen):
                nearest = seen[np.abs((np.arange(n)[:, None] - seen[None, :] + n // 2) % n - n // 2).argmin(axis=1)]
                uv = uv[nearest]
                if image is not None:
                    # A speck of odd paint on the edge would run up the whole strip: each ray takes, from itself
                    # and its two neighbours either side, the paint closest to the usual colour round there.
                    colour = sample_texture(image, uv)
                    usual = ndimage.median_filter(colour, size=(FIT_SPREAD, 1), mode="wrap")
                    window = (np.arange(n)[:, None] + np.arange(-2, 3)[None, :]) % n
                    best = np.linalg.norm(colour[window] - usual[:, None, :], axis=2).argmin(axis=1)
                    uv = uv[window[np.arange(n), best]]
            else:
                uv = None
        here, there = np.arange(n), (np.arange(n) + 1) % n
        points, paints, tris = [], [], []
        first = len(v)
        for k in range(len(rows) - 1):
            # Each piece has its own four points, so two rays whose paint sits far apart in the picture (another
            # texture island) do not drag the paint between them across the picture.
            start = first + sum(len(p) for p in points)
            points += [ring_points[k][here], ring_points[k][there], ring_points[k + 1][here], ring_points[k + 1][there]]
            if uv is not None:
                close = np.linalg.norm(uv[there] - uv[here], axis=1) < FIT_UV_NEAR
                beside = np.where(close[:, None], uv[there], uv[here])
                paints += [uv[here], beside, uv[here], beside]
            lh, lt, uh, ut = (start + i * n + here for i in range(4))
            tris += [np.column_stack([uh, lh, lt]), np.column_stack([uh, lt, ut])]
        v = np.vstack([v, *points])
        faces = np.vstack([body.faces, *tris])
        added = int(sum(len(t) for t in tris))
        if uv is not None and getattr(visual, "uv", None) is not None:
            visual.uv = np.vstack([visual.uv, *paints])
    out = trimesh.Trimesh(v, faces, visual=visual, process=False)
    lowest = float(body.vertices[touched, 1].min()) if touched.any() else low
    return out, {"rim": rim, "edge": low + rise, "lowest": lowest, "room": room, "painted": painted,
                 "moved_points": int(touched.sum()), "added_faces": added,
                 "widest_move": float(shift.max()), "cloth_move": cloth, "depth": deep,
                 "taper": float(np.abs(rim - foot).max() / deep) if deep > 0 else 0.0}


def fit_neck(body, head, cut: float, ring: float, head_height: float, body_share: float | None = None,
             body_depth: float | None = None, shape: bool = True):
    """Give the new HEAD's neck the shape of BODY's where they meet: (body, head, report).

    The new head is sized and placed by its skull, so its neck comes out a little slimmer or thicker than the
    body's and stands further forward or back, and it can end above the body's: a step or a gap all round.
    Both necks are cut straight where each is still a whole ring (the body's just under CUT, the head's the
    lowest it has, as `bridge_neck` does). Then the head's neck is pulled onto the body's: its points move
    straight out from (or in to) the neck's centre line by the difference between the two rings, and down by
    the gap between them, in full at the head's edge and less and less above it, up to FIT_REACH of
    HEAD_HEIGHT. Along each ray the change ends below the first thing that is not neck (the jaw, an ear,
    hair), so the face and the skull keep their shape. The head's last strip ends a little down inside the
    body's neck, so the two edges overlap and do not have to meet. Paint moves with the points, and the
    head's neck paint takes the body's colour at the edge, fading out over the same height, so the two skins
    do not meet in a line. Where the body's edge is a collar that touches the neck (far wider than the new
    neck, or painted far from its skin) the head's neck and the body's stump behind it are left as they are.
    BODY_SHARE (0..1, default FIT_BODY) lets the body's neck stump take that share of the difference first
    (`fit_stump`): the base of its neck, collar and all, shrinks or grows towards the head's ring as one smooth
    field (down to BODY_DEPTH of HEAD_HEIGHT, default FIT_DEPTH) and grows up towards the head's edge, and the
    head's neck then moves only for what is left. The two paints then meet half-way instead of the head taking
    the body's.
    A ring that is partly under a collar is fitted on its bare part only: behind the collar neither neck moves
    and nothing is cut, with a smooth change-over beside it; with more than FIT_COVERED of the ring covered
    there is nothing to aim at and nothing changes. SHAPE False leaves every point where it is and only
    brings the head's neck paint to the body's at the seam (for two necks that already agree in shape).
    RING is the neck's radius. Nothing changes, and the report says why, when either neck has no whole ring
    near the cut or too few rays have room to change."""
    axis = neck_centre(head, head_height, ring)
    reach, step = BRIDGE_REACH * ring, BRIDGE_STEP * head_height
    span, tuck = BRIDGE_SEARCH * head_height, FIT_TUCK * head_height
    limit, least = FIT_REACH * head_height, FIT_LEAST * head_height
    angles = fit_angles()

    def find(mesh, heights, missing):
        for y in heights:
            radius = fit_ring(mesh, float(y), axis, reach, missing)
            if radius is not None:
                return float(y), radius
        return None, None

    low, foot = find(body, np.arange(cut - step, cut - span, -step), BRIDGE_MISSING)
    if low is None:
        return body, head, {"skipped": "the body's neck is not a whole ring under the cut"}
    # The head's edge: its lowest ring without holes, one step of it to spare for the strip inside the body.
    start = max(low - step, float(head.vertices[:, 1].min()) + 0.2 * step)
    edge, _ = find(head, np.arange(start, low + limit, step), 0.0)
    if edge is None:
        return body, head, {"skipped": "the new head has no whole ring of neck near the cut"}
    seat = edge + step                                  # this height of the head lands on the body's edge
    neck = fit_ring(head, seat, axis, reach, 0.0)
    if neck is None:
        return body, head, {"skipped": "the new head has no whole ring of neck near the cut"}
    # A collar that touches the neck reads as neck in the body's ring. It is cloth where its paint is far from
    # the head's neck skin, or where the ring is far wider than the neck; the neck does not follow it there.
    mine = ring_paint(head, seat + step * np.arange(3), axis, angles)
    theirs = ring_paint(body, low - step * np.arange(3), axis, angles)
    collar = foot > FIT_COLLAR * neck
    gap = None
    if mine is not None and theirs is not None:
        gap = theirs - mine
        with np.errstate(invalid="ignore", divide="ignore"):
            hue = theirs / theirs.sum(axis=1, keepdims=True) - mine / mine.sum(axis=1, keepdims=True)
            collar |= (np.abs(gap).max(axis=1) > FIT_CLOTH) & ((np.abs(hue).max(axis=1) > FIT_HUE)
                                                              | (foot > FIT_LOOSE * neck))
    cloth = collar.copy()
    collar = ndimage.maximum_filter1d(collar.astype(float), FIT_SPREAD, mode="wrap")
    if collar.mean() > FIT_COVERED:
        return body, head, {"skipped": "a collar stands round most of the neck at the cut", "expected": True}
    free = 1 - ndimage.uniform_filter1d(collar, FIT_SPREAD, mode="wrap")
    foot = neck + free * (foot - neck)
    change, drop = foot - neck, seat - low
    if np.abs(change).max() > FIT_MOST * neck.mean():
        return body, head, {"skipped": "the two necks differ too much to be the same neck"}

    # Where the change ends along each ray: under the first slice that holds more than neck.
    top = np.full(len(angles), seat + limit)
    for y in np.arange(seat + limit, seat, -step):
        far = ring_far(ring_segments(head, float(y), axis)[0], angles)
        with np.errstate(invalid="ignore"):
            top[far > FIT_CHIN * neck] = y - FIT_CLEAR * head_height
    top = ndimage.uniform_filter1d(ndimage.minimum_filter1d(top, FIT_SPREAD, mode="wrap"), FIT_SPREAD, mode="wrap")
    room = float((top - seat >= least).mean())
    if room < FIT_SHARE:
        return body, head, {"skipped": "too little neck between the cut and the jaw to change its shape"}
    top = np.maximum(top, seat + least)

    # The body's neck under its edge can lean in: the head's last strip stays inside the narrowest of it.
    inside = foot.copy()
    for y in (low - tuck / 2, low - tuck):
        inside = np.fmin(inside, ring_reach(ring_segments(body, float(y), axis)[0], angles))
    inside = BRIDGE_INSET * inside

    # Paint: the head's neck takes the body's skin colour at the edge (evened out round the neck: ray by ray
    # it comes out striped).
    shade = None
    if gap is not None:
        known = ~np.isnan(gap).any(axis=1) & (collar == 0)
        if known.mean() >= FIT_SHARE:
            gap = np.stack([np.interp(angles, angles[known], gap[known, k], period=2 * np.pi) for k in range(3)], axis=1)
            shade = ndimage.uniform_filter1d(gap, FIT_PAINT_SPREAD, axis=0, mode="wrap") * free[:, None]

    body_share = float(np.clip(FIT_BODY if body_share is None else body_share, 0.0, 1.0))
    if shape:
        # The body's neck above its edge goes, except behind a collar, where it is what closes the collar.
        body = cut_neck(body, low, axis, np.interp(bridge_angles(), angles, np.where(
            collar > 0, 0.0, (1 + BRIDGE_WALL / 2) * foot), period=2 * np.pi))
    else:
        body_share, change, drop = 0.0, np.zeros(len(angles)), 0.0
    stump, body_low = None, low
    if body_share > 0:
        # The body's stump goes first; the head's neck then fits what the body has become.
        rise = body_share * drop if (body_share * drop >= 2 * step and not collar.any()) else 0.0
        both = None
        if shade is not None:
            room_ok = ndimage.uniform_filter1d(
                (stump_room(body, low, foot, axis, head_height) >= FIT_STUMP * head_height).astype(float),
                FIT_SPREAD, mode="wrap")
            both = -0.5 * room_ok[:, None] * shade               # the body's paint comes half-way where it can,
            shade = shade + both                                 # and the head's paint goes the rest
        body, stump = fit_stump(body, low, foot, foot + body_share * free * (neck - foot), rise, axis, head_height,
                                both, body_depth, bool(collar.any()))
        if both is not None and not stump["painted"]:
            shade = shade - both
        low = stump["edge"]
        reached = fit_ring(body, low - 0.2 * step, axis, reach)
        foot = stump["rim"] if reached is None else neck + free * (reached - neck)
        change, drop = foot - neck, seat - low
        inside = foot.copy()
        for y in (low - tuck / 2, low - tuck):
            inside = np.fmin(inside, ring_reach(ring_segments(body, float(y), axis)[0], angles))
        inside = BRIDGE_INSET * inside
    if shape:
        # Under a collar the head keeps everything it has below its edge, as it was.
        head = split_at(cut_neck(head, edge, axis, np.interp(bridge_angles(), angles, np.where(
            collar > 0, 0.0, reach), period=2 * np.pi), above=False), seat)
    v = np.array(head.vertices, float)
    at = ring_angle(v, axis)

    def along(values):
        return np.interp(at, angles, values, period=2 * np.pi)

    off = v[:, [0, 2]] - axis
    radius = np.hypot(off[:, 0], off[:, 1])
    v_top, v_neck = along(top), along(neck)
    is_neck = (v[:, 1] < v_top) & (radius <= FIT_CHIN * v_neck)
    share = np.clip((v_top - v[:, 1]) / (v_top - seat), 0.0, 1.0)
    weight = np.where(is_neck, share * share * (3 - 2 * share), 0.0)
    sunk = np.where(is_neck & shape, np.clip((seat - v[:, 1]) / (seat - edge), 0.0, 1.0), 0.0)
    moved = radius + weight * along(change)
    under = sunk > 0
    moved[under] = np.minimum(moved, along(foot) + sunk * (along(inside) - along(foot)))[under]
    moved = np.maximum(moved, 0.0)
    scale = np.divide(moved, radius, out=np.ones_like(radius), where=radius > 1e-9)
    height = np.where(under, low - sunk * tuck, v[:, 1] - weight * drop)
    bare = along(free)
    if (bare < 1).any():
        # Beside and under a collar the head's neck goes only part of the way, and not at all behind it.
        part = bare < 1
        scale[part] = 1 + bare[part] * (scale[part] - 1)
        height[part] = v[part, 1] + bare[part] * (height[part] - v[part, 1])
    touched = (np.abs(scale - 1) > 0) | (np.abs(height - v[:, 1]) > 0)
    if shape:
        v[touched] = np.column_stack([axis[0] + off[touched, 0] * scale[touched], height[touched],
                                      axis[1] + off[touched, 1] * scale[touched]])
    fitted = head.copy()
    fitted.vertices = v
    painted = False
    capped = False
    if shade is not None:
        # A big difference is not skin tone: on a short neck the body's paint under the cut is the shadow beside
        # its collar, and the new chin took it as a beard (pc_f01e, 0.27). The head goes this far and no further.
        capped = bool(np.abs(shade).max() > FIT_SHADE)
        shade = np.clip(shade, -FIT_SHADE, FIT_SHADE)
        tint = np.maximum(weight, (sunk > 0).astype(float))[:, None] * np.stack(
            [along(shade[:, k]) for k in range(3)], axis=1)
        picture = tint_paint(head, tint)
        if picture is not None:
            import trimesh

            material = head.visual.material.copy()
            material.baseColorTexture = picture
            fitted.visual = trimesh.visual.TextureVisuals(uv=head.visual.uv, material=material)
            painted = True
    return body, fitted, {"body_at": round(body_low, 4), "seam_at": round(low, 4), "head_at": round(seat, 4), "top": round(float(top.max()), 4),
                          "top_front": round(float(np.interp(0.0, angles, top)), 4),
                          "reach_share": round(float((top.max() - seat) / head_height), 3),
                          "rays_with_room": round(room, 3),
                          "moved_down": round(float(drop), 4),
                          "widest_move": round(float(np.abs(change).max()), 4),
                          "mean_move": round(float(change.mean()), 4),
                          "moved_points": int(touched.sum()), "collar_share": round(float(collar.mean()), 3),
                          "paint_matched": painted,
                          "paint_change": None if shade is None else round(float(np.abs(shade).max()), 3),
                          "paint_capped": capped,
                          "paint_stretch_front": round(float(1 + drop / (np.interp(0.0, angles, top) - seat)), 2),
                          "body_share": body_share, "head_height": round(float(head_height), 4), "shape": bool(shape),
                          "cloth_rays": "".join("c" if c else "." for c in cloth[::len(angles) // BRIDGE_RAYS * 2 or 1]),
                          "covered": [int(c) for c in np.interp(bridge_angles(), angles, collar, period=2 * np.pi) > 0],
                          "body": None if stump is None else {
                              "lowest": round(stump["lowest"], 4), "grown_up": round(stump["edge"] - body_low, 4),
                              "room_front": round(float(np.interp(0.0, angles, stump["room"])), 4),
                              "room_back": round(float(stump["room"][0]), 4),
                              "room_sides": [round(float(np.interp(a, angles, stump["room"])), 4) for a in (-np.pi / 2, np.pi / 2)],
                              "room_least": round(float(stump["room"].min()), 4),
                              "widest_move": round(stump["widest_move"], 4), "taper": round(stump["taper"], 2),
                              "cloth_move": round(stump["cloth_move"], 4), "depth": round(stump["depth"], 4),
                              "moved_points": stump["moved_points"], "added_faces": stump["added_faces"],
                              "paint_matched": stump["painted"]},
                          "centre": [round(float(c), 4) for c in axis]}


def seam_normals(parts, height: float, axis, reach: float, covered=None) -> list[np.ndarray]:
    """The normals of PARTS (a list of (vertices, normals)) with one shared set along the ring at HEIGHT.

    The head and the body are separate parts with their own normals, so light breaks along the line where
    their edges meet although the surface runs on. Every outward-facing point within REACH of HEIGHT takes
    the mean normal of all such points round the same part of the ring (from every part). Points facing the
    centre line (a model's inner wall) are left alone, and so is the part of the ring that is COVERED (true
    or false for each of `bridge_angles`: a collar, where the two edges do not meet). The parts are not welded."""
    rays = bridge_angles()
    width = rays[1] - rays[0]
    total = np.zeros((len(rays), 3))
    picked = []
    for vertices, normals in parts:
        off = vertices[:, [0, 2]] - np.asarray(axis, float)
        near = np.abs(vertices[:, 1] - height) <= reach
        outward = (off * normals[:, [0, 2]]).sum(axis=1) > 0
        index = np.flatnonzero(near & outward)
        slot = np.round((ring_angle(vertices[index], axis) - rays[0]) / width).astype(int) % len(rays)
        np.add.at(total, slot, normals[index])
        picked.append((index, slot))
    total = total + 0.5 * (np.roll(total, 1, axis=0) + np.roll(total, -1, axis=0))
    length = np.linalg.norm(total, axis=1, keepdims=True)
    mean = np.divide(total, length, out=np.zeros_like(total), where=length > 1e-12)
    out = []
    for (vertices, normals), (index, slot) in zip(parts, picked):
        new = np.array(normals, float)
        good = np.linalg.norm(mean[slot], axis=1) > 0
        if covered is not None:
            good &= ~np.asarray(covered, bool)[slot]
        new[index[good]] = mean[slot[good]]
        out.append(new)
    return out


def fit_seams(fit: dict | None, head_height: float) -> list[tuple]:
    """(height, centre line, reach, covered) of each ring where a neck fit (`fit_neck`'s report) left two edges
    meeting, for `seam_normals`; empty when nothing was fitted."""
    if not fit or "skipped" in fit:
        return []
    reach = 0.5 * BRIDGE_STEP * head_height
    heights = sorted({fit["seam_at"], fit["body_at"]})
    return [(float(y), fit["centre"], reach, fit.get("covered")) for y in heights]


def fit_notes(fit: dict | None) -> list[str]:
    """What a person should check about a neck fit (`fit_neck`'s report); empty when it went well."""
    if fit and "skipped" in fit and not fit.get("expected"):
        return [f"the new neck was not fitted to the body's: {fit['skipped']}"]
    return []


def bridge_notes(bridge: dict | None) -> list[str]:
    """What a person should check about a neck bridge (`bridge_neck`'s report); empty when it went well."""
    if not bridge:
        return []
    if "skipped" in bridge:
        return [] if bridge.get("expected") else [f"the neck seam was not bridged: {bridge['skipped']}"]
    if bridge["height_share"] > BRIDGE_TALL:
        return [f"the neck bridge is tall ({bridge['height_share']:.0%} of the head's height): check the neck"]
    return []


OWN_REACH = 0.3           # own collar: the body's collar is looked for up to this share of the head's height above
OWN_BELOW = 0.15          # the cut, starting from the cloth this share of the head's height under it,
OWN_WIDE = 2.5            # within this many neck radii of the neck's centre line
OWN_SKIN = 0.35           # each model's skin colour: its face, from this share of the head's height above the cut
OWN_OVER = 0.1            # the new head's copy of the collar may stand this share of the head's height above the body's
OWN_PATCH = 12            # fewer cloth-coloured triangles than this in one patch of the new head are paint, not a collar
OWN_LEAST = 0.1           # a collar that rises less than this share of the head's height above the cut is left to the fit
OWN_RAYS = 48             # the collar's inner wall is measured in this many directions round the neck
OWN_NAPE = 0.12           # the old nape kept behind the collar, as a liner under the new one, shrinks this much
OWN_ALONG = 0.8           # (points at least this share of the wall's distance from the neck's centre line)
OWN_GROW = 1              # the body keeps this many rings of its old skin along the collar: its rim stays whole
OWN_LAP = 0.02            # the new head stops short where its skin ends more than this share of its height over the cut
OWN_TUCK = 0.015          # the new head's skin ends this share of the head's height under the cut, straight
OWN_CORE = 0.05           # the neck's middle: that of the old skin up to this share of the head's height over the cut


def vertex_paint(mesh) -> np.ndarray | None:
    """MESH's paint (0..1, RGB) at each vertex, None without a texture or UVs."""
    image = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    uv = getattr(mesh.visual, "uv", None)
    if image is None or uv is None or len(uv) != len(mesh.vertices):
        return None
    return sample_texture(image, np.asarray(uv, float))


def cloth_paint(paint: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Which PAINT colours are cloth, not SKIN: far from it in some channel (`FIT_CLOTH`) and in hue too
    (`FIT_HUE`, each channel's share of the colour), so shaded skin and a beard stay skin."""
    hue = paint / np.maximum(paint.sum(axis=1, keepdims=True), 1e-6) - skin / max(float(skin.sum()), 1e-6)
    return (np.abs(paint - skin).max(axis=1) > FIT_CLOTH) & (np.abs(hue).max(axis=1) > FIT_HUE)


def joined(mesh) -> np.ndarray:
    """One number per vertex of MESH, the same for vertices at the same place (a glTF mesh is split at every
    texture seam)."""
    return np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)[1].reshape(-1)


def skin_middle(points: np.ndarray) -> np.ndarray | None:
    """The middle (x, z) of a band of neck skin POINTS (x, y, z): half-way between its two sides, each way. None
    with too few points to tell."""
    if len(points) < MIN_SLICE_POINTS:
        return None
    flat = points[:, [0, 2]]
    return (np.percentile(flat, 2, axis=0) + np.percentile(flat, 98, axis=0)) / 2


def stops_short(heights: np.ndarray, rays: np.ndarray, level: float) -> np.ndarray:
    """For each of OWN_RAYS directions round the neck: does the new head's skin (its points' HEIGHTS, each in
    direction RAYS) end above LEVEL there? A direction with no skin at all stops short too."""
    floor = np.full(OWN_RAYS, np.inf)
    for k in range(OWN_RAYS):
        if (rays == k).sum() >= 3:
            floor[k] = np.percentile(heights[rays == k], 5)
    return floor > level


def end_straight(mesh, low: float):
    """MESH without what hangs under the height LOW, cut along it in a straight line."""
    return clip_below(mesh, mesh.vertices[:, 1] < low, low)


def own_collar(full, real: np.ndarray, head, cut: float, axis: np.ndarray, ring: float, head_height: float):
    """(which of FULL's vertices are its collar above CUT, which of HEAD's triangles are the new head's copy of
    that collar, a report), or (None, None, why not). The report's "nape" marks FULL's old skin to keep as a
    liner (`graft` takes it out of the report).

    On a short neck the jaw hangs below a stand collar's rim, so the cut under the jaw sits inside the collar.
    The plane then took the part of the body's collar the rig gave to the head and left the rest in shreds, and
    the new head, whole above the cut, brought the collar of its crop with it: a second collar over the first,
    ending in the ragged hem the crop's edge left (pc_m01b). The body's collar is the good one. It is the cloth
    (by paint, against the face's skin) that joins the cloth under the cut; the new head's copy is the head's
    cloth in that colour up to a little above the body's rim."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    body_paint, head_paint = vertex_paint(full), vertex_paint(head)
    if body_paint is None or head_paint is None:
        return None, None, {"skipped": "a model has no paint to tell cloth from skin"}
    v, w = full.vertices, head.vertices
    face = cut + OWN_SKIN * head_height

    def skin_of(points, paint, among):
        front = among & (points[:, 1] > face) & (points[:, 2] > axis[1])
        return np.median(paint[front], axis=0) if front.sum() >= MIN_SLICE_POINTS else None

    body_skin, head_skin = skin_of(v, body_paint, real), skin_of(w, head_paint, np.ones(len(w), bool))
    if body_skin is None or head_skin is None:
        return None, None, {"skipped": "no face found to take the skin colour from"}
    # Everything below is measured round the neck, so from its middle: the graft's own axis (`neck_core`) can sit
    # a centimetre to one side (pc_m03d), and then one side of the new neck counted as outside the collar and went,
    # and the old skin kept as a liner stood in the collar's open front. The old neck's skin just above the cut.
    core = real & ~cloth_paint(body_paint, body_skin) & (v[:, 1] >= cut) & (v[:, 1] < cut + OWN_CORE * head_height)
    middle = skin_middle(v[core])
    moved = None if middle is None else [round(float(d), 4) for d in middle - axis]
    axis = axis if middle is None else middle
    near = np.hypot(v[:, 0] - axis[0], v[:, 2] - axis[1]) < OWN_WIDE * ring
    top = cut + OWN_REACH * head_height
    cloth = cloth_paint(body_paint, body_skin) & near & (v[:, 1] > cut - OWN_BELOW * head_height) & (v[:, 1] < top)
    # The body's cloth as one sheet: flood from the cloth under the cut that is not the head's, through cloth.
    spot = joined(full)
    edges = np.vstack([full.faces[:, [0, 1]], full.faces[:, [1, 2]], full.faces[:, [2, 0]]])
    edges = edges[cloth[edges].all(axis=1)]
    count = int(spot.max()) + 1
    graph = coo_matrix((np.ones(len(edges)), (spot[edges[:, 0]], spot[edges[:, 1]])), shape=(count, count))
    label = connected_components(graph, directed=False)[1][spot]
    seeds = np.unique(label[cloth & ~real & (v[:, 1] < cut)])
    collar = cloth & np.isin(label, seeds)
    lifted = collar & (v[:, 1] >= cut)
    if lifted.sum() < MIN_SLICE_POINTS:
        return None, None, {"skipped": "the body has no collar above the cut", "expected": True}
    rim = float(np.percentile(v[lifted, 1], 98))
    if rim - cut < OWN_LEAST * head_height:      # pc_f03: a neckline that touches the neck just above the cut
        return None, None, {"skipped": "the body's collar ends at the cut", "expected": True}
    colour = np.median(body_paint[lifted], axis=0)
    # What the new head has outside the collar's inner wall, under the rim, is no neck either (the skin its crop
    # had under the collar: it came out as flaps on the collar's back).
    def turn(points):
        angle = np.arctan2(points[:, 0] - axis[0], points[:, 2] - axis[1])
        return np.floor((angle + np.pi) / (2 * np.pi) * OWN_RAYS).astype(int) % OWN_RAYS

    def far(points):
        return np.hypot(points[:, 0] - axis[0], points[:, 2] - axis[1])

    wall = np.full(OWN_RAYS, np.inf)
    ray, reach = turn(v[lifted]), far(v[lifted])
    for k in range(OWN_RAYS):
        if (ray == k).sum() >= 3:
            wall[k] = np.percentile(reach[ray == k], 10)
    outside = (w[:, 1] < rim) & (far(w) > wall[turn(w)]) & (w[:, 2] < axis[1])   # behind only: in front and at the
    #                                                          sides the jaw hangs over the collar, further out than its wall
    # Behind the collar the new head's nape ends where its copy of the collar began, often above the body's rim,
    # and the gap showed the inside. The body's old nape stays there, a little smaller, as a liner (back half
    # only: in front the old chin hangs in the same band).
    nape = real & ~lifted & (v[:, 1] >= cut) & (v[:, 1] < rim + OWN_OVER * head_height) \
        & (v[:, 2] < axis[1]) & np.isfinite(wall[turn(v)])
    # The triangles between the collar and the old skin go whole (`clip_below`), and where the paint of a rim
    # point is already skin that tore the rim. A ring of old skin along the collar, inside it, keeps it whole.
    all_edges = np.vstack([full.faces[:, [0, 1]], full.faces[:, [1, 2]], full.faces[:, [2, 0]]])
    sheet = lifted.copy()
    for _ in range(OWN_GROW):
        on = np.zeros(count, bool)
        on[spot[lifted]] = True
        touch = on[spot[all_edges]].any(axis=1)
        ringed = np.zeros(count, bool)
        ringed[spot[all_edges[touch]].ravel()] = True
        # (Only skin that lies along the wall and under the rim: in a coarse mesh the next point can be the jaw.)
        lifted = lifted | (ringed[spot] & real & (v[:, 1] >= cut) & (v[:, 1] < rim) & (far(v) > OWN_ALONG * wall[turn(v)]))
    # The new head's copy: its cloth (against its own skin) up to a little above the rim, in patches that hold
    # enough of the collar's colour. Whole patches: the copy's pale rim and dark folds are nearer skin than the
    # collar in colour, and stayed behind as scraps under the ear.
    import trimesh

    low = cloth_paint(head_paint, head_skin) & (w[:, 1] < rim + OWN_OVER * head_height)
    like = low & (np.linalg.norm(head_paint - colour, axis=1) < np.linalg.norm(head_paint - head_skin, axis=1))
    copy = low[head.faces].sum(axis=1) >= 2
    flaps = outside[head.faces].all(axis=1)
    if copy.any():
        pairs = trimesh.graph.face_adjacency(joined(head)[head.faces])
        pairs = pairs[copy[pairs].all(axis=1)]
        patch = trimesh.graph.connected_component_labels(pairs, node_count=len(head.faces))
        seeds = copy & (like[head.faces].sum(axis=1) >= 2)
        sizes = np.bincount(patch[seeds], minlength=int(patch.max()) + 1)
        copy &= sizes[patch] >= OWN_PATCH
    copy |= flaps
    # The liner and the ring of old skin are for where the new head stops short of the cut once its copy of the
    # collar is gone. Where its own skin comes down to the cut (beside the opening of a fold-down collar, pc_m03d)
    # they stood next to the new neck as ragged scraps of the old one.
    skin = np.zeros(len(w), bool)
    skin[np.unique(head.faces[~copy])] = True
    skin &= w[:, 1] < rim + OWN_OVER * head_height
    short = stops_short(w[skin, 1], turn(w[skin]), cut + OWN_LAP * head_height)[turn(v)]
    nape &= short
    lifted = sheet | (lifted & short)
    lifted = lifted & ~nape
    report = {"rim_above_cut": round(rim - cut, 4), "over": round(OWN_OVER * head_height, 4),
              "body_collar_points": int(lifted.sum()),
              "kept_from_the_cut": int((lifted & real).sum()), "head_copy_faces": int(copy.sum()),
              "head_flap_faces": int(flaps.sum()), "centre_moved": moved,
              "nape_liner_points": int(nape.sum()), "nape": nape,
              "collar_colour": [round(float(c), 3) for c in colour]}
    return lifted, copy, report


def drop_low_pieces(mesh, top: float):
    """MESH without the loose pieces that lie wholly under TOP, whatever their size (the largest piece always
    stays): what hung on the new head's copy of the collar and came free when the copy went."""
    import trimesh

    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    label = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(welded.faces))
    sizes = np.bincount(label)
    if len(sizes) < 2:
        return mesh
    high = np.zeros(len(sizes))
    np.maximum.at(high, label, mesh.vertices[mesh.faces][:, :, 1].max(axis=1))
    gone = (high < top) & (np.arange(len(sizes)) != sizes.argmax())
    return mesh.submesh([np.flatnonzero(~gone[label])], append=True)


def own_notes(own: dict | None) -> list[str]:
    """What a person should glance at after the collar split."""
    if not own or own.get("expected"):
        return []
    if "skipped" in own:
        return [f"the body's own collar was not kept apart from the new head's: {own['skipped']}"]
    return []


def graft(full, head, full_box, head_box, crop, head_width, bone, shape, grow=None, liner=LINER_SHRINK,
          cut_at=None, skull_fit=False, neck_bridge=False, neck_fit=False, neck_fit_body=None,
          neck_fit_depth=None, neck_fit_shape=True, keep_collar=False):
    """Swap FULL's head region for HEAD (both trimesh meshes; HEAD is moved in place).

    CUT_AT: "neck" finds the neck (`neck_height`) and swaps only above it; a height (model
    units) cuts there, e.g. one a person marked; None swaps the whole head region.
    NECK_BRIDGE closes the seam between the two necks (`bridge_neck`) when the cut sits under the
    jaw (a bare neck or a tight collar); a wide collar hides its seam and is left alone.
    NECK_FIT, same case, first gives the new neck the body's shape at the cut (`fit_neck`); the seam is then
    closed by the two overlapping edges and the bridge is not needed. If the fit cannot be done the bridge runs.
    NECK_FIT_BODY (0..1, default `FIT_BODY`) is the share of that the body's neck stump takes before the head's.
    NECK_FIT_SHAPE False keeps both necks' shapes and only matches the paint and the light across the seam.
    KEEP_COLLAR, same case: where the cut sits inside a stand collar, the body keeps all of its collar and
    the new head loses its copy of it (`own_collar`).

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
    own = copy = None
    if keep_collar:
        if tight:
            stays, copy, own = own_collar(full, real, head, cut, axis, ring, float(np.ptp(full.vertices[real, 1])))
            if stays is not None:
                nape = own.pop("nape")
                swap &= ~(stays | nape)
        else:
            own = {"skipped": "the cut is not under the jaw (a wide collar, a marked height or the whole head)",
                   "expected": True}
    keep &= in_head
    body_mesh = full.copy()
    body_mesh.vertices, gone = liner_split(full.vertices, swap, liner)
    if copy is not None and nape.any():
        moved = body_mesh.vertices.copy()
        moved[np.ix_(nape, [0, 2])] = axis + (1 - OWN_NAPE) * (moved[np.ix_(nape, [0, 2])] - axis)
        body_mesh.vertices = moved
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
        body = clip_below(body_mesh, gone, cut, above_whole=copy is not None)
    else:
        drop = gone[full.faces].any(axis=1) | liner_miss
        body = body_mesh.submesh([np.flatnonzero(~drop)], append=True)
    kept = keep[head.faces].all(axis=1)
    if copy is not None:
        kept &= ~copy
    part = head.submesh([np.flatnonzero(kept)], append=True)
    if copy is not None:
        part = drop_low_pieces(part, cut + own["rim_above_cut"] + own["over"])
        # An open collar shows the throat: there the head's skin hung on under the cut, over the body's own, and
        # ended in the ragged edge its copy of the collar left (pc_m03d). The body's throat is whole from the cut
        # down, so the head ends just under it, in a straight line.
        low = cut - OWN_TUCK * float(np.ptp(full.vertices[real, 1]))
        part = end_straight(part, low)
    pushed = 0.0
    if tight:
        part = drop_scraps(part, cut, axis, ring)
        # After the scraps are picked (they are judged by distance from the neck, which this changes). The
        # whole body, not only the head bone's share: that starts above the cut, on the skull. One straight
        # slide back: the head keeps its shape.
        # (Measured on the head with its copy of the collar still on: without it the head has no back there.)
        whole = part.vertices if copy is None else head.vertices[np.unique(head.faces[keep[head.faces].all(axis=1)])]
        pushed = neck_back_gap(whole, full.vertices, cut, axis, ring,
                               NECK_BACK_SPAN * float(np.ptp(full.vertices[real, 1])))
        part.vertices = part.vertices - [0.0, 0.0, pushed]
    bridge = fit = None
    hidden = {"skipped": "the body's own collar stands round the seam", "expected": True} if copy is not None else None
    if neck_fit:
        if hidden:      # a fit or a bridge would cut or bend the collar that was just kept
            fit = hidden
        elif tight:
            body, part, fit = fit_neck(body, part, cut, ring, float(np.ptp(full.vertices[real, 1])), neck_fit_body,
                                        neck_fit_depth, neck_fit_shape)
        else:
            fit = {"skipped": "the cut is not under the jaw (a wide collar, a marked height or the whole head)"}
    if neck_bridge and fit is not None and "skipped" not in fit:
        bridge = {"skipped": "the new neck was fitted to the body's (no seam left to bridge)", "expected": True}
    elif neck_bridge:
        if hidden:
            bridge = hidden
        elif tight:
            body, part, bridge = bridge_neck(body, part, cut, ring, float(np.ptp(full.vertices[real, 1])))
        else:
            bridge = {"skipped": "the cut is not under the jaw (a wide collar, a marked height or the whole head)",
                      "expected": True}
    report = {"head_bones": picked, "outline_margin_px": int(grow), "liner_shrink": liner,
              "liner_tucked": tucked, "liner_dropped_faces": int(liner_miss.sum()),
              "cut_at": None if cut is None else round(cut, 4),
              "neck_ring": None if ring is None else round(ring, 4),
              "slid_back": round(pushed, 4), "neck_fit": fit, "neck_bridge": bridge,
              **({"own_collar": own} if keep_collar else {}),
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
