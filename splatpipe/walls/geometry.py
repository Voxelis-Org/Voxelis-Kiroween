"""The Walls_Step's geometry, as pure numpy functions in float64 (walls spec, Req 6).

The bas-relief change. A Bas_Relief_Change (a, b, c, d) changes a depth map z into z′ by

    1/z′ = a/z + m × (b·x + c·y + d)

at each pixel, with m the Median_Inverse_Depth of the map it changes and (x, y) the pixel's
Normalised_Coordinates: Kriegman and Belhumeur's perspective bas-relief family, written relative
to the depth it changes. The Identity_Change (1, 0, 0, 0) changes nothing.

- normalised_coords: each pixel's (x, y) = ((u − cx) / fx, (v − cy) / fy), with pixel (i, j) of a
  w × h grid centred at (u, v) = ((i + 0.5) × W / w, (j + 0.5) × H / h) in the Shared_Camera's
  W × H pixels, the Prior_Format's rule, so a grid of any size, the Score_Size or the Prior_Size,
  is described by the same rays. rays gives each pixel's Ray r = (x, y, 1).
- median_inverse_depth: m, the median of 1/z over the Covered_Pixels, or over a Facet's pixels.
- apply_change (Req 6.1, 6.2): the new inverse depth, a × inverse + m × (b·x + c·y + d), worked out
  exactly so, so the Identity_Change gives back every inverse depth bit for bit: 1.0 × ι is ι, and
  m × (0·x + 0·y + 0) is 0.
- depth_factor (Req 6.3): s = z′ / z = 1 / (a + m·z·(b·x + c·y + d)), the old inverse depth over
  the new one.
- compose (Req 6.5): the first change on a map of median m₁ and the second on the result, of
  median m₂, as the one change (a₁·a₂, a₂·b₁ + (m₂/m₁)·b₂, a₂·c₁ + (m₂/m₁)·c₂, a₂·d₁ + (m₂/m₁)·d₂)
  of the first map. With Lₖ = bₖ·x + cₖ·y + dₖ: ι₂ = a₂·ι₁ + m₂·L₂ = a₁a₂·ι₀ + m₁(a₂·L₁ +
  (m₂/m₁)·L₂).
- changed_normal (Req 6.4, D4): a·n + m·z·(n·r)·(b, c, d), set to unit length and turned round
  when its dot product with r is above 0, so it faces the camera. On a plane n·X = δ, a point
  z·r has z(n·r) = δ, so ι = (n·r)/δ is affine in (x, y). The change keeps it affine, so the
  changed points lie on one plane, and this is that plane's normal.

A pixel with no new depth or no new normal is NaN, in every part of a normal. A new inverse depth
that isn't finite and above 0 gives no new depth (Req 6.3), and then no Depth_Factor and no
Changed_Normal either. A raw normal of length 0, or with a part that isn't finite, gives no new
normal (Req 6.4). depth_factor and changed_normal decide whether a pixel has a new depth by the
same test, the sign and finiteness of a + m·z·(b·x + c·y + d), which is z × the new inverse depth.

Normals are in the camera frame, OpenCV's axes as COLMAP, gsplat and MoGe use them: x right,
y down, z forward. A normal facing the camera has n·r below 0.

The projection and the warp (Req 9.3, 14.2). An Image_Pose is the (4, 4) world-to-camera matrix
[[R, t], [0, 0, 0, 1]] that splatpipe.train.scene.world_to_camera builds, and a K is the (3, 3)
Shared_Camera that splatpipe.train.scene.scaled_intrinsics scales to a size, in COLMAP's pixel
convention, with the top-left pixel's centre at (0.5, 0.5).
- back_project: a pixel's camera point z·(x, y, 1).
- project: where a camera point lands with a K, u = fx·X/Z + cx and v = fy·Y/Z + cy, and its z.
- warp_maps (Req 9.3): each Score_Size pixel's point at s·z on its Ray in the Prior_Frame's
  camera, C = s·z·(x, y, 1), goes to the world, W = Rₚᵀ(C − tₚ), and into the Source_Frame's
  camera, C_s = R_s·W + t_s, which the Source_Frame's K at the Score_Size projects to (u, v).
  It gives OpenCV's sample coordinates (u − 0.5, v − 0.5), for cv2.remap, C_s's z, the depth
  the warp predicts in the Source_Frame, and where that z is above 0 and (u, v) lies on
  [0, W) × [0, H) of the Source_Frame.
- zbuffer_project (Req 14.2, A11): points in a frame's camera, each landing on a pixel of a
  w × h grid by the Depth_Step's pixel rule (min(⌊u·w/W⌋, w − 1), min(⌊v·h/H⌋, h − 1)), keep the
  nearest depth per pixel, and a pixel no point lands on stays unfilled.
The poses are applied to the points elementwise, one part at a time, never with @ or np.dot.

Every function is elementwise over the pixels, with numpy's own loops and no BLAS call, so the
results are the same bits whatever the thread count (C14). This module imports only the standard
library, numpy and splatpipe.train.scene.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from splatpipe.train.scene import Camera

Change = Sequence[float]  # (a, b, c, d)


def normalised_coords(camera: Camera, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's Normalised_Coordinates on a `width` × `height` grid, as two (height, width)
    float64 arrays x and y.

    Pixel (i, j) is centred at u = (i + 0.5) × W / width and v = (j + 0.5) × H / height in the
    Shared_Camera's W × H pixels, the Prior_Format's rule, and x = (u − cx) / fx, y = (v − cy) / fy.
    """
    u = (np.arange(width, dtype=np.float64) + 0.5) * camera.width / width
    v = (np.arange(height, dtype=np.float64) + 0.5) * camera.height / height
    x_row = (u - camera.cx) / camera.fx
    y_column = (v - camera.cy) / camera.fy
    x = np.broadcast_to(x_row[np.newaxis, :], (height, width)).copy()
    y = np.broadcast_to(y_column[:, np.newaxis], (height, width)).copy()
    return x, y


def rays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each pixel's Ray r = (x, y, 1), as (..., 3) float64."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def median_inverse_depth(depth: np.ndarray, covered: np.ndarray) -> float:
    """m, the median of 1 / `depth` over the pixels where `covered` is true.

    The caller gives finite depths above 0 at those pixels. Refuses a map with no such pixel,
    because a change needs an m.
    """
    covered = np.asarray(covered, dtype=bool)
    if not covered.any():
        raise ValueError("median_inverse_depth: no covered pixel")
    inverse = 1.0 / np.asarray(depth, dtype=np.float64)[covered]
    return float(np.median(inverse))


def apply_change(inverse_depth: np.ndarray, change: Change, m: float, x: np.ndarray,
                 y: np.ndarray) -> np.ndarray:
    """The new inverse depth a × inverse + m × (b·x + c·y + d), in float64 (Req 6.1).

    Worked out exactly in that order, so the Identity_Change gives back every inverse depth bit
    for bit (Req 6.2). A pixel whose new inverse depth isn't finite and above 0 is NaN: it gets no
    new depth (Req 6.3).
    """
    a, b, c, d = (float(value) for value in change)
    inverse = np.asarray(inverse_depth, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        new = a * inverse + float(m) * (b * x + c * y + d)
    return np.where(np.isfinite(new) & (new > 0), new, np.nan)


def _denominator(z: np.ndarray, change: Change, m: float, x: np.ndarray,
                 y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a + m·z·(b·x + c·y + d), which is z × the new inverse depth, and where the pixel gets a new
    depth: z finite and above 0, and the denominator finite and above 0."""
    a, b, c, d = (float(value) for value in change)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = a + float(m) * z * (b * x + c * y + d)
        valid = (np.isfinite(z) & (z > 0) & np.isfinite(denominator) & (denominator > 0))
    return denominator, valid


def depth_factor(z: np.ndarray, change: Change, m: float, x: np.ndarray,
                 y: np.ndarray) -> np.ndarray:
    """s = z′ / z = 1 / (a + m·z·(b·x + c·y + d)), the old inverse depth over the new one, in
    float64 (Req 6.3). NaN at a pixel with no new depth, or whose s isn't finite."""
    denominator, valid = _denominator(z, change, m, x, y)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        factor = 1.0 / denominator
    return np.where(valid & np.isfinite(factor), factor, np.nan)


def compose(first: Change, second: Change, m1: float, m2: float) -> tuple[float, float, float, float]:
    """The one change of the first depth map that equals `first` on it, of median `m1`, then
    `second` on the result, of median `m2` (Req 6.5):
    (a₁·a₂, a₂·b₁ + (m₂/m₁)·b₂, a₂·c₁ + (m₂/m₁)·c₂, a₂·d₁ + (m₂/m₁)·d₂), relative to m = m₁."""
    a1, b1, c1, d1 = (float(value) for value in first)
    a2, b2, c2, d2 = (float(value) for value in second)
    ratio = float(m2) / float(m1)
    return (a1 * a2, a2 * b1 + ratio * b2, a2 * c1 + ratio * c2, a2 * d1 + ratio * d2)


def changed_normal(normal: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray,
                   change: Change, m: float) -> np.ndarray:
    """Each pixel's Changed_Normal, as (..., 3) float64 (Req 6.4, D4).

    The raw normal a·n + m·z·(n·r)·(b, c, d), from the pixel's normal n, its depth z and its Ray
    r = (x, y, 1), set to unit length and negated when its dot product with r is above 0, so it
    faces the camera. NaN in every part at a pixel with no new depth, or whose raw normal has a
    length of 0 or a part that isn't finite.
    """
    a, b, c, d = (float(value) for value in change)
    normal = np.asarray(normal, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny, nz = normal[..., 0], normal[..., 1], normal[..., 2]
    _, has_depth = _denominator(z, change, m, x, y)
    with np.errstate(over="ignore", invalid="ignore"):
        along = float(m) * z * (nx * x + ny * y + nz)  # m·z·(n·r)
        raw_x = a * nx + along * b
        raw_y = a * ny + along * c
        raw_z = a * nz + along * d
        length = np.hypot(np.hypot(raw_x, raw_y), raw_z)  # no overflow on the way
        finite = np.isfinite(raw_x) & np.isfinite(raw_y) & np.isfinite(raw_z)
        good = has_depth & finite & np.isfinite(length) & (length > 0)
        unit_x = raw_x / length
        unit_y = raw_y / length
        unit_z = raw_z / length
        facing = np.where(unit_x * x + unit_y * y + unit_z > 0, -1.0, 1.0)
    result = np.stack([unit_x * facing, unit_y * facing, unit_z * facing], axis=-1)
    return np.where(good[..., np.newaxis], result, np.nan)


def back_project(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each pixel's camera point z·(x, y, 1), as (..., 3) float64, from its depth z and its
    Normalised_Coordinates (x, y)."""
    z, x, y = np.broadcast_arrays(np.asarray(z, dtype=np.float64),
                                  np.asarray(x, dtype=np.float64),
                                  np.asarray(y, dtype=np.float64))
    return np.stack([z * x, z * y, z], axis=-1)


def _to_world(pose: np.ndarray, px: np.ndarray, py: np.ndarray,
              pz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera points (px, py, pz) in the world, W = Rᵀ(P − t) for the world-to-camera `pose`,
    one part at a time: Wⱼ = Σᵢ R[i, j]·(Pᵢ − tᵢ), with no BLAS call (C14)."""
    r = np.asarray(pose, dtype=np.float64)[:3, :3]
    t = np.asarray(pose, dtype=np.float64)[:3, 3]
    dx, dy, dz = px - t[0], py - t[1], pz - t[2]
    return (r[0, 0] * dx + r[1, 0] * dy + r[2, 0] * dz,
            r[0, 1] * dx + r[1, 1] * dy + r[2, 1] * dz,
            r[0, 2] * dx + r[1, 2] * dy + r[2, 2] * dz)


def _to_camera(pose: np.ndarray, wx: np.ndarray, wy: np.ndarray,
               wz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points (wx, wy, wz) in the camera of the world-to-camera `pose`, C = R·W + t, one
    part at a time: Cᵢ = Σⱼ R[i, j]·Wⱼ + tᵢ, with no BLAS call (C14)."""
    r = np.asarray(pose, dtype=np.float64)[:3, :3]
    t = np.asarray(pose, dtype=np.float64)[:3, 3]
    return (r[0, 0] * wx + r[0, 1] * wy + r[0, 2] * wz + t[0],
            r[1, 0] * wx + r[1, 1] * wy + r[1, 2] * wz + t[1],
            r[2, 0] * wx + r[2, 1] * wy + r[2, 2] * wz + t[2])


def _project_parts(k: np.ndarray, px: np.ndarray, py: np.ndarray,
                   pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) = (fx·X/Z + cx, fy·Y/Z + cy) with the K `k`. A point at Z = 0 gives an infinity or
    NaN, with no warning, which every caller's test of Z > 0 leaves out."""
    k = np.asarray(k, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        u = k[0, 0] * px / pz + k[0, 2]
        v = k[1, 1] * py / pz + k[1, 2]
    return u, v


def project(points_camera: np.ndarray,
            k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each camera point of `points_camera`, (..., 3), lands with the K `k`, as (u, v, z):
    u = fx·X/Z + cx and v = fy·Y/Z + cy in the pixels of the size `k` is scaled to, in COLMAP's
    convention, and z = Z, its depth. Nothing is dropped: a caller keeps the points with z above 0
    and (u, v) inside its image."""
    points = np.asarray(points_camera, dtype=np.float64)
    px, py, pz = points[..., 0], points[..., 1], points[..., 2]
    u, v = _project_parts(k, px, py, pz)
    return u, v, pz.copy()


def warp_maps(prior_depth: np.ndarray, factor: float | np.ndarray, prior_pose: np.ndarray,
              source_pose: np.ndarray, source_k: np.ndarray, x: np.ndarray, y: np.ndarray, *,
              source_size: tuple[int, int] | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The warp of a Source_Frame into a Prior_Frame at the Depth_Factor `factor` (Req 9.3), as
    (map_x, map_y, source_z, inside), each float64 or bool in the pixels' shape.

    Each pixel's point at `factor` × `prior_depth` on its Ray (x, y, 1), in the Prior_Frame's
    camera, goes to the world through `prior_pose` and into the Source_Frame's camera through
    `source_pose`, both world-to-camera Image_Poses from world_to_camera, applied elementwise
    (C14). `source_k` is the Shared_Camera at the Score_Size from scaled_intrinsics, which
    projects the point to (u, v). map_x = u − 0.5 and map_y = v − 0.5 are OpenCV's sample
    coordinates for cv2.remap, source_z is the depth the warp predicts in the Source_Frame, and
    `inside` is true where source_z is finite and above 0 and 0 ≤ u < W, 0 ≤ v < H.

    `source_size` is the Source_Frame's (W, H). It defaults to the size of `prior_depth`, (its
    last axis, the one before), because the Prior_Frame and the Source_Frame are both at the
    Score_Size. A pixel with no depth (NaN) is never inside.
    """
    z = np.asarray(prior_depth, dtype=np.float64) * np.asarray(factor, dtype=np.float64)
    z, x, y = np.broadcast_arrays(z, np.asarray(x, dtype=np.float64),
                                  np.asarray(y, dtype=np.float64))
    if source_size is None:
        if z.ndim < 2:
            raise ValueError("warp_maps: give source_size for depths that aren't an image")
        source_size = (z.shape[-1], z.shape[-2])
    width, height = source_size
    with np.errstate(over="ignore", invalid="ignore"):
        world = _to_world(prior_pose, z * x, z * y, z)
        source_x, source_y, source_z = _to_camera(source_pose, *world)
        u, v = _project_parts(source_k, source_x, source_y, source_z)
        inside = (np.isfinite(source_z) & (source_z > 0)
                  & (u >= 0) & (u < width) & (v >= 0) & (v < height))
        return u - 0.5, v - 0.5, source_z, inside


def zbuffer_project(points_camera: np.ndarray, k: np.ndarray, width: int, height: int, *,
                    camera_size: tuple[int, int] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The nearest depth per pixel of a `width` × `height` grid of the camera points
    `points_camera`, (..., 3), already in the frame's camera, as (depth, filled): (height, width)
    float64, NaN where no point lands, and (height, width) bool (Req 14.2, A11).

    `k` is the Shared_Camera at `camera_size`, its (W, H), from scaled_intrinsics, or the
    Shared_Camera itself at its full size. A point lands at (u, v) = project(...) and goes on
    when its z is finite and above 0 and 0 ≤ u < W, 0 ≤ v < H. Its pixel is the Depth_Step's
    rule, (min(⌊u·w/W⌋, w − 1), min(⌊v·h/H⌋, h − 1)). The landing (pixel, z) pairs are sorted with
    np.lexsort((z, pixel)) and each pixel keeps its first, the smallest z. `camera_size` defaults
    to (`width`, `height`), for a `k` already scaled to the grid.
    """
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    full_width, full_height = (width, height) if camera_size is None else camera_size
    u, v = _project_parts(k, points[:, 0], points[:, 1], points[:, 2])
    z = points[:, 2]
    with np.errstate(invalid="ignore"):
        lands = (np.isfinite(z) & (z > 0)
                 & (u >= 0) & (u < full_width) & (v >= 0) & (v < full_height))
    column = np.minimum(np.floor(u[lands] * width / full_width).astype(np.int64), width - 1)
    row = np.minimum(np.floor(v[lands] * height / full_height).astype(np.int64), height - 1)
    pixel = row * width + column
    order = np.lexsort((z[lands], pixel))
    pixel, nearest = pixel[order], z[lands][order]
    first = np.ones(pixel.shape, dtype=bool)
    first[1:] = pixel[1:] != pixel[:-1]
    depth = np.full(width * height, np.nan)
    filled = np.zeros(width * height, dtype=bool)
    depth[pixel[first]] = nearest[first]
    filled[pixel[first]] = True
    return depth.reshape(height, width), filled.reshape(height, width)
