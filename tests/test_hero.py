"""The hero route's maths: finding the head, cropping it, putting it back.

Each of these broke once while the route was being worked out by hand (2026-10-21): a fixed
grey background made the whole picture "figure", a guessed scale shrank the head 12%, an
outline made of corner points dropped triangles as holes, a fixed margin trimmed the hair.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import hero


def test_picture_box_prefers_the_cut_out_over_grey():
    # A darker graded background (the elf's ran 202-222) is not the studio's light grey.
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[..., :3] = 210
    rgba[20:80, 30:60] = (90, 120, 90, 255)
    assert hero.picture_box(Image.fromarray(rgba, "RGBA")) == (30, 59, 20, 79)
    # Without a cut-out the same picture reads as all figure: the failure the alpha path fixes.
    assert hero.picture_box(Image.fromarray(rgba[..., :3], "RGB")) == (0, 99, 0, 99)


def test_picture_box_falls_back_to_light_grey():
    rgb = np.full((50, 50, 3), 236, np.uint8)
    rgb[10:40, 5:15] = 40
    assert hero.picture_box(Image.fromarray(rgb, "RGB")) == (5, 14, 10, 39)


def test_project_maps_the_mesh_onto_its_picture_box():
    verts = np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0]])
    fr = hero.frame(verts, (100, 300, 50, 450))      # 400 px tall for 4 units
    px, py = hero.project(verts, fr)
    assert np.allclose(px, [100, 300]) and np.allclose(py, [450, 50])   # picture y runs down


def test_head_bones_take_the_top_region_and_a_neck_inside_it():
    heights = np.concatenate([np.linspace(0.9, 1.0, 50),    # head
                              np.linspace(0.88, 0.95, 20),  # neck, middle inside the head's span
                              np.linspace(0.6, 0.9, 50)])   # chest
    bone = np.array([5] * 50 + [4] * 20 + [3] * 50)
    assert hero.head_bones(bone, heights) == [4, 5]


def test_head_bones_leave_a_neck_below_the_head():
    heights = np.concatenate([np.linspace(0.9, 1.0, 50), np.linspace(0.8, 0.86, 20)])
    bone = np.array([5] * 50 + [4] * 20)
    assert hero.head_bones(bone, heights) == [5]


def test_crop_box_is_square_padded_and_inside_the_picture():
    px, py = np.array([460.0, 558.0]), np.array([27.0, 168.0])   # the warrior's head pixels
    x0, y0, x1, y1 = hero.crop_box(px, py, 1024, 1024)
    assert x1 - x0 == y1 - y0
    assert x0 < 460 and x1 > 558 and y1 > 168
    assert y0 >= 0                       # a head at the top edge is not cropped off the picture
    assert (x1 - x0) == pytest.approx(141 * hero.CROP_PAD, abs=1)


def test_crop_to_full_puts_a_crop_pixel_back_where_it_came_from():
    full = np.array([[-1.0, -2.0, -0.5], [1.0, 2.0, 0.5]])
    full_frame = hero.frame(full, (100, 300, 50, 450))
    head = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    head_frame = hero.frame(head, (100, 900, 100, 900))          # head picture 1024 px
    crop = (150, 50, 250, 150)                                   # 100 px of the full picture
    tf = hero.crop_to_full(full_frame, head_frame, crop, 1024)
    moved = (np.c_[head, np.ones(2)] @ tf.T)[:, :3]
    px, py = hero.project(moved, full_frame)
    # head picture box (100..900) of 1024 -> full pixels 150 + p * 100/1024
    assert np.allclose(px, [150 + 100 * 100 / 1024, 150 + 900 * 100 / 1024])
    assert np.allclose(py, [50 + 900 * 100 / 1024, 50 + 100 * 100 / 1024])


def test_place_on_matches_centre_and_height_only():
    real = np.array([[0.0, 1.0, 0.0], [0.2, 1.4, 0.1]])
    new = np.array([[5.0, 5.0, 5.0], [5.4, 5.8, 5.2]])
    tf = hero.place_on(real, new)
    moved = (np.c_[new, np.ones(2)] @ tf.T)[:, :3]
    assert np.ptp(moved[:, 1]) == pytest.approx(0.4)
    assert np.allclose(moved.mean(axis=0), real.mean(axis=0))


def test_silhouette_fills_whole_triangles_not_just_corners():
    # One big triangle: its corners are three pixels, its inside must be filled too.
    px, py = np.array([10.0, 90.0, 50.0]), np.array([90.0, 90.0, 10.0])
    mask = hero.silhouette(np.array([[0, 1, 2]]), px, py, (100, 100))
    assert mask[60, 50] and mask.sum() > 2500
    assert not mask[5, 5]


def test_liner_keeps_the_old_head_s_back_half_shrunk_and_drops_its_face():
    verts = np.array([[0, 2, 1.0], [0, 2, -1.0], [0, 0, 0.5]])   # face (+Z), back of head, chest
    real = np.array([True, True, False])
    moved, gone = hero.liner_split(verts, real, 0.1)
    assert gone.tolist() == [True, False, False]      # old face goes; back of head stays as liner
    assert np.allclose(moved[1], [0, 2, -0.9])        # shrunk toward the head's centre (0, 2, 0)
    assert np.allclose(moved[2], verts[2])            # the rest of the body is untouched


def test_liner_off_drops_the_whole_old_head():
    verts = np.array([[0, 2, 1.0], [0, 2, -1.0], [0, 0, 0.5]])
    real = np.array([True, True, False])
    moved, gone = hero.liner_split(verts, real, 0.0)
    assert gone.tolist() == [True, True, False] and np.allclose(moved, verts)


def _ring(height, width, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a) * width / 2, np.full(n, height), np.sin(a) * width / 2], axis=1)


def _profile(rows):
    """A head region stacked from rings: (height, width) per row."""
    return np.concatenate([_ring(h, w) for h, w in rows])


def test_neck_height_cuts_between_the_collar_and_the_skull():
    # The warrior's profile (2026-10-22): stray bits, a collar, the narrower jaw, the head. The
    # neck hides inside the collar, so the narrow band above it is the chin: cutting in the middle
    # of it left the old body's mouth showing under the new face. Cut at the collar's top.
    rows = ([(0.862 + 0.002 * i, 0.022) for i in range(3)]
            + [(0.882 + 0.002 * i, 0.090) for i in range(6)]      # collar, top at 0.892
            + [(0.898 + 0.002 * i, 0.064) for i in range(5)]      # chin and jaw
            + [(0.910 + 0.003 * i, 0.078) for i in range(25)])    # head
    pts = _profile(rows)
    cut = hero.neck_height(pts[:, 1], pts[:, 0])
    assert 0.890 <= cut <= 0.899            # at the collar's top, below the chin


def test_neck_height_is_none_without_a_collar():
    # The region ends at a bare neck (the elf): nothing below is wider, so cut nothing extra.
    rows = ([(0.88 + 0.002 * i, 0.050) for i in range(8)]
            + [(0.896 + 0.003 * i, 0.075) for i in range(25)])
    pts = _profile(rows)
    assert hero.neck_height(pts[:, 1], pts[:, 0]) is None


def test_neck_height_ignores_slices_with_too_few_points():
    rows = ([(0.882 + 0.002 * i, 0.090) for i in range(6)]
            + [(0.898 + 0.002 * i, 0.064) for i in range(5)]
            + [(0.910 + 0.003 * i, 0.078) for i in range(25)])
    pts = _profile(rows)
    stray = np.array([[0.0, 0.87, 0.0], [0.001, 0.871, 0.0]])  # a few points below, very narrow
    cut = hero.neck_height(np.r_[pts[:, 1], stray[:, 1]], np.r_[pts[:, 0], stray[:, 0]])
    assert 0.890 <= cut <= 0.899


def test_neck_ring_is_the_skull_s_radius_round_its_centre():
    pts = np.concatenate([_ring(0.95 + 0.005 * i, 0.094) for i in range(6)])     # skull, radius 0.047
    pts[:, 0] += 0.01                                                             # off the origin
    real = np.ones(len(pts), bool)
    axis, radius = hero.neck_ring(pts, real, cut=0.9)
    assert np.allclose(axis, [0.01, 0.0], atol=1e-3) and abs(radius - 0.047) < 0.002


def test_head_cut_splits_the_collar_band_by_the_ring_not_by_height():
    # The warrior (2026-10-22): the new head sits a little lower, so in the collar band the body's
    # old chin (inside the ring) and its collar (outside) share heights with the new mouth.
    # x = distance from the neck's centre line; ring 0.047; collar top 0.90.
    body = np.array([[0.0, 0.95, 0.04],    # old face above the collar: goes
                     [0.0, 0.87, 0.03],    # old chin inside the ring: goes (it showed through)
                     [0.0, 0.87, 0.07],    # its own collar: stays
                     [0.0, 0.80, 0.03]])   # chest, not head region: stays
    real = np.array([True, True, True, False])
    head = np.array([[0.0, 0.95, 0.04],    # new face: stays
                     [0.0, 0.88, 0.03],    # new mouth and chin inside the ring: stay
                     [0.0, 0.87, 0.07]])   # its scuffed collar: goes
    swap, keep = hero.head_cut(body, real, head, cut=0.90, axis=np.zeros(2), ring=0.047)
    assert swap.tolist() == [True, True, False, False]
    assert keep.tolist() == [True, True, False]


def test_head_cut_off_swaps_the_whole_region():
    body = np.array([[0, 0.95, 0.04], [0, 0.87, 0.07], [0, 0.80, 0.03]])
    real = np.array([True, True, False])
    swap, keep = hero.head_cut(body, real, np.array([[0, 0.95, 0.0], [0, 0.85, 0.08]]),
                               cut=None, axis=np.zeros(2), ring=0.047)
    assert swap.tolist() == [True, True, False] and keep.tolist() == [True, True]


def test_paired_points_pair_only_spots_within_reach():
    head = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 5, 5]])
    body = np.array([[0.01, 0, 0], [1.02, 0, 0]])
    head_idx, body_idx = hero.paired_points(head, body, reach=0.05)
    assert head_idx.tolist() == [0, 1] and body_idx.tolist() == [0, 1]   # the far point is left out


def test_match_to_body_lands_an_all_metal_head_on_the_body_s_level():
    # The long-haired woman: Pixal3D called her whole head metal; her body said cloth and skin.
    mr = Image.fromarray(np.dstack([np.zeros((4, 4)), np.full((4, 4), 254), np.full((4, 4), 255)]).astype(np.uint8))
    head_seen = np.tile([0.0, 254 / 255, 1.0], (10, 1))
    body_seen = np.tile([0.0, 0.98, 0.0], (10, 1))
    fixed, report = hero.match_to_body(mr, head_seen, body_seen, channels=(1, 2))
    a = np.asarray(fixed).astype(float) / 255
    assert a[..., 2].max() == 0                         # no metal left
    assert a[..., 1].mean() == pytest.approx(0.98, abs=0.005)
    assert report["big_change"]


def test_match_to_body_moves_level_and_contrast_but_keeps_the_pattern():
    head = np.array([[0.2, 0.4], [0.6, 0.8]])           # a darker, low-contrast head
    image = Image.fromarray(np.round(np.dstack([head] * 3) * 255).astype(np.uint8))
    head_seen = np.repeat(head.reshape(-1, 1), 3, axis=1)
    body_seen = head_seen * 1.2 + 0.05                   # the body: brighter, more contrast
    fixed, report = hero.match_to_body(image, head_seen, body_seen)
    a = np.asarray(fixed).astype(float)[..., 0] / 255
    assert np.allclose(a.mean(), body_seen[:, 0].mean(), atol=0.01)
    assert np.all(np.diff(a.reshape(-1)) > 0)            # the order of light and dark is kept
    assert report["r"]["stretch"] == pytest.approx(1.2, abs=0.01) and not report["big_change"]


def test_sample_texture_reads_glTF_uvs_with_v_up():
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8))
    got = hero.sample_texture(image, np.array([[0.0, 1.0], [0.0, 0.0]]))   # top-left, bottom-left
    assert np.allclose(got[0], [1, 0, 0]) and np.allclose(got[1], [0, 0, 1])


def _skinned_glb(positions: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> bytes:
    blobs = [positions.astype(np.float32).tobytes(), joints.astype(np.uint8).tobytes(),
             weights.astype(np.float32).tobytes()]
    views, offset = [], 0
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        offset += len(blob)
    n = len(positions)
    document = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": offset}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5121, "count": n, "type": "VEC4"},
                      {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC4"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
    }
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    binary = b"".join(blobs)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_skinned_vertices_take_each_vertex_s_strongest_joint():
    positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], float)
    joints = np.array([[1, 2, 0, 0], [3, 1, 0, 0], [0, 4, 0, 0]])
    weights = np.array([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0], [0.4, 0.6, 0, 0]])
    got_positions, got_joints = hero.skinned_vertices(_skinned_glb(positions, joints, weights))
    assert np.allclose(got_positions, positions)
    assert got_joints.tolist() == [1, 1, 4]


def test_regions_survive_a_rescaled_re_export():
    # The rigger re-exports the mesh at another scale and place; boxing both to one height lines them up.
    verts = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], float)
    rig = verts * 3 + 10
    assert hero.regions(verts, rig, np.array([7, 8, 9])).tolist() == [7, 8, 9]


def _sphere(radius, at=(0.0, 0.0, 0.0), subdivisions=2):
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh.apply_translation(at)
    return mesh


def test_tuck_liner_pulls_an_oversized_old_head_inside_the_new_one():
    # The space marine (2026-11-09): the body's old head carried more volume than the new one and
    # had ears the new head lacks, so the liner's ears poked out beside the new ears as a second
    # set. Ray from the new head's centre: anything outside is pulled to just inside the surface.
    import trimesh

    new_head = _sphere(1.0)
    old = trimesh.util.concatenate([_sphere(1.45), _sphere(0.3, (1.15, 0.1, 0.0), subdivisions=1)])
    verts = np.vstack([old.vertices, [[0.0, -2.0, 0.0]]])   # the old head, then a shoulder point
    liner = np.r_[np.ones(len(old.vertices), bool), False]

    moved, drop = hero.tuck_liner(verts, old.faces, liner, new_head, cut=-0.85,
                                  axis=np.array([0.0, 0.0]), radius=1.0)

    old_head = moved[:len(old.vertices)]
    beside = (old.vertices[:, 1] > -0.85) & (np.abs(old.vertices[:, 0]) > np.abs(old.vertices[:, 2]))
    assert np.linalg.norm(old_head[beside], axis=1).max() <= 1.0 + 1e-6  # no second ear
    assert np.allclose(old_head[~beside], old.vertices[~beside])         # the back, neck and collar stay
    assert not drop.any()                                               # every ray hit the new head
    assert np.allclose(moved[-1], [0.0, -2.0, 0.0])                     # the shoulder is untouched


def test_tuck_liner_leaves_the_back_of_the_liner_alone():
    # The liner is what fills the gap under the hair at the back. Pulling it in there showed the
    # new head's own neck through the gap, so only what shows beside the head is pulled in.
    new_head = _sphere(1.0)
    verts = np.array([[0.0, 0.2, -1.4]])        # behind the head and outside it
    moved, _ = hero.tuck_liner(verts, np.zeros((0, 3), int), np.ones(1, bool), new_head,
                               cut=-0.85, axis=np.array([0.0, 0.0]), radius=1.0)
    assert np.allclose(moved, verts)


def test_tuck_liner_drops_faces_with_no_new_head_behind_them():
    # A new head is cut open at the neck. Liner points below its jaw have no head behind them:
    # nothing can hide them, so their faces go rather than showing old skin.
    new_head = _sphere(1.0)
    faces = new_head.faces
    new_head.update_faces(new_head.vertices[faces][:, :, 1].min(axis=1) > 0.1)  # open at the neck
    new_head.remove_unreferenced_vertices()

    verts = np.array([[0.6, -0.5, 0.0], [0.9, -0.5, 0.0], [0.9, -0.5, 0.2], [0.6, -0.5, 0.2]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    moved, drop = hero.tuck_liner(verts, faces, np.ones(4, bool), new_head, cut=-0.9,
                                  axis=np.array([0.0, 0.0]), radius=1.0)

    assert drop.tolist() == [True, True]
    assert np.allclose(moved, verts)                              # nothing to pull them to


def test_drop_scraps_keeps_small_pieces_inside_the_neck_and_drops_flaps_outside():
    import trimesh

    from image_to_3dlab.hero import drop_scraps
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    head.apply_translation([0, 0.4, 0])
    skin = trimesh.creation.cylinder(radius=0.03, height=0.06)
    skin.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    skin.apply_translation([0, 0.3, 0])
    flap = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
    flap.apply_translation([0, 0.29, 0.08])
    out = drop_scraps(trimesh.util.concatenate([head, skin, flap]), 0.31, np.array([0.0, 0.0]), 0.045)
    assert len(out.faces) == len(head.faces) + len(skin.faces)
    out = drop_scraps(trimesh.util.concatenate([head, skin, flap]), 0.31)
    assert len(out.faces) == len(head.faces)


def test_neck_back_gap_measures_how_far_the_new_neck_stands_in_front_of_the_old_one():
    from image_to_3dlab.hero import neck_back_gap
    ys = np.linspace(0.30, 0.40, 161)
    old = np.array([[0.0, y, -0.04 if y < 0.35 else -0.03] for y in ys])          # old neck back, then skull
    new = np.array([[0.0, y, -0.025 if y < 0.35 else -0.03] for y in ys])         # new neck 1.5 cm forward
    assert neck_back_gap(new, old, 0.30, np.array([0.0, 0.0]), 0.04, 0.1) == pytest.approx(0.015, abs=1e-3)
    assert neck_back_gap(old, new, 0.30, np.array([0.0, 0.0]), 0.04, 0.1) == 0.0   # never slides forward


def test_clip_below_cuts_crossing_triangles_on_the_plane_and_keeps_the_rest():
    import trimesh

    from image_to_3dlab.hero import clip_below
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2]])
    uv = v[:, :2] / 2
    m = trimesh.Trimesh(v, f, visual=trimesh.visual.TextureVisuals(uv=uv), process=False)
    out = clip_below(m, v[:, 1] >= 1.0, 1.0)
    assert out.vertices[:, 1].max() == 1.0                       # nothing above the plane, edge on it
    assert np.isclose(out.area, 1.0)                             # exactly the lower half of the 1 x 2 square
    assert np.allclose(out.visual.uv, out.vertices[:, :2] / 2)   # UVs follow the new points
    assert len(out.split(only_watertight=False)) == 1            # still one piece: new points are shared


def test_drop_scraps_treats_texture_seams_as_joined():
    import trimesh

    from image_to_3dlab.hero import drop_scraps
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    head.apply_translation([0, 0.4, 0])
    neck = trimesh.creation.cylinder(radius=0.03, height=0.2, sections=100)   # 400 faces: kept whole, not halves
    neck.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    neck.apply_translation([0, 0.3, 0.0])
    # the neck's lower half as its own patch (own copies of the seam points), as a glTF export has it
    lower = neck.faces[neck.triangles_center[:, 1] < 0.3]
    upper = neck.faces[neck.triangles_center[:, 1] >= 0.3]
    split = trimesh.util.concatenate([trimesh.Trimesh(neck.vertices, upper, process=False),
                                      trimesh.Trimesh(neck.vertices, lower, process=False)])
    whole = trimesh.util.concatenate([head, split])
    assert len(drop_scraps(whole, 0.25).faces) == len(whole.faces)


def _tube(radius, bottom, top, ragged=None, sections=48, rings=40, seed=0):
    """An open tube round the y axis with UVs; RAGGED (lo, hi) drops half the triangles between those heights."""
    import trimesh
    angle = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ys = np.linspace(bottom, top, rings + 1)
    v = np.array([[radius * np.sin(a), y, radius * np.cos(a)] for y in ys for a in angle])
    uv = np.array([[a / (2 * np.pi), (y - bottom) / (top - bottom)] for y in ys for a in angle])
    f = []
    for r in range(rings):
        for k in range(sections):
            a, b = r * sections + k, r * sections + (k + 1) % sections
            f += [(a, b, a + sections), (b, b + sections, a + sections)]
    f = np.array(f)
    if ragged:
        mid = v[f][:, :, 1].mean(axis=1)
        torn = (mid > ragged[0]) & (mid < ragged[1]) & (np.random.default_rng(seed).random(len(f)) < 0.5)
        f = f[~torn]
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), (200, 150, 120)))
    return trimesh.Trimesh(v, f, visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)


def _radius(mesh):
    return np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 2])


def test_ring_reach_takes_the_neck_skin_not_a_collar_and_says_where_nothing_is():
    from image_to_3dlab.hero import ring_reach

    def square(half):
        c = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], float)
        return np.stack([c, np.roll(c, -1, axis=0)], axis=1)

    walls = np.concatenate([square(0.9), square(1.0), square(1.5), square(1.6)])   # neck (two walls), collar
    assert ring_reach(walls, np.array([0.0, np.pi / 2])) == pytest.approx([1.0, 1.0])
    one_side = ring_reach(square(1.0)[:1], np.array([0.0, np.pi]))                  # the segment at z = -1 only
    assert np.isnan(one_side[0]) and one_side[1] == pytest.approx(1.0)
    ends = np.zeros((16, 2, 2))
    ends[4:8, :, 0] = 0.25                                                           # the skin's UVs
    reach, uv = ring_reach(walls, np.array([0.0]), ends)
    assert reach == pytest.approx([1.0]) and uv[0] == pytest.approx([0.25, 0.0])


def test_bridge_neck_cuts_both_ragged_necks_straight_and_closes_the_gap():
    from image_to_3dlab.hero import bridge_neck, bridge_ring, neck_centre
    body = _tube(0.040, 0.0, 0.32, ragged=(0.30, 0.33))
    head = _tube(0.035, 0.29, 0.50, ragged=(0.28, 0.31), seed=1)
    new_body, new_head, report = bridge_neck(body, head, cut=0.30, ring=0.04, head_height=0.2)
    low, high = report["body_at"], report["head_at"]
    assert 0.29 < low < 0.30 and 0.31 <= high < 0.32 and "skipped" not in report
    assert new_body.vertices[:, 1].max() == pytest.approx(low, abs=1e-4)             # the stump is gone, cut straight
    old = new_head.vertices[:len(new_head.vertices) - 4 * 72]
    assert old[:, 1].min() == pytest.approx(high, abs=1e-4)                          # so is the head's ragged edge
    axis = neck_centre(new_head, 0.2)
    for y in np.linspace(low + 1e-3, high - 1e-3, 5):                                # between them: a whole ring
        ring = bridge_ring(new_head, float(y), axis, 0.064, missing=0.0)
        assert ring is not None and 0.03 < ring.min() and ring.max() < 0.04          # inside both necks
    assert new_head.vertices[:, 1].min() < low                                       # tucked into the body
    assert len(new_head.visual.uv) == len(new_head.vertices)
    assert len(new_head.faces) == len(new_head.faces[(new_head.faces < len(new_head.vertices)).all(axis=1)])


def test_bridge_neck_leaves_a_collar_standing():
    import trimesh

    from image_to_3dlab.hero import bridge_neck
    neck = _tube(0.040, 0.0, 0.32, ragged=(0.30, 0.33))
    collar = _tube(0.060, 0.20, 0.33)
    body = trimesh.util.concatenate([neck, collar])
    head = _tube(0.035, 0.29, 0.50, ragged=(0.28, 0.31), seed=1)
    new_body, _, report = bridge_neck(body, head, cut=0.30, ring=0.04, head_height=0.2)
    assert "skipped" not in report
    above = new_body.vertices[new_body.vertices[:, 1] > report["body_at"] + 1e-4]
    assert len(above) and np.hypot(above[:, 0], above[:, 2]).min() > 0.055           # only the collar is left up there
    assert above[:, 1].max() == pytest.approx(0.33)


def test_bridge_neck_changes_nothing_when_there_is_no_seam_to_close():
    from image_to_3dlab.hero import bridge_neck, bridge_notes
    body = _tube(0.040, 0.0, 0.32, ragged=(0.30, 0.33))
    deep = _tube(0.035, 0.25, 0.50)                                                  # the head's neck runs into the body
    same_body, same_head, report = bridge_neck(body, deep, cut=0.30, ring=0.04, head_height=0.2)
    assert same_body is body and same_head is deep and report["expected"] and bridge_notes(report) == []
    torn = _tube(0.040, 0.0, 0.32, ragged=(0.0, 0.33))                               # no whole ring on the body
    head = _tube(0.035, 0.29, 0.50, ragged=(0.28, 0.31), seed=1)
    same_body, same_head, report = bridge_neck(torn, head, cut=0.30, ring=0.04, head_height=0.2)
    assert same_body is torn and same_head is head
    assert bridge_notes(report) == ["the neck seam was not bridged: the body's neck is not a whole ring under the cut"]


def test_bridge_notes_flags_a_tall_bridge_only():
    from image_to_3dlab.hero import bridge_notes
    assert bridge_notes(None) == [] and bridge_notes({"height_share": 0.05}) == []
    assert "tall" in bridge_notes({"height_share": 0.2})[0]
