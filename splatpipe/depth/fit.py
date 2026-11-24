"""The Depth_Step's pure maths: the Field_Of_View, the Fit_Points, the two-pass Depth_Fit, a
Fitted_Frame's Prior arrays and the job's Metres_Per_Unit (depth-priors spec, Req 4.3, 4.7, 5 and
6).

Every function works on the arrays and numbers it's given, reads no file and imports no torch, so
the Test_Suite tests each rule on tiny made-up data on a laptop (Req 5.10).

- field_of_view (Req 4.3): degrees(2 atan(W / (2 fx))), from the Shared_Camera's full width W and
  its fx, both in pixels. Scaling the Frame doesn't change it.
- geometry_pixels (Req 4.7): the Geometry_Pixels, where MoGe's mask is true and its depth is
  finite and above 0.
- fit_points (Req 5.1, A2): one Frame's Fit_Points. Each Observed_Point p goes into the camera
  frame with the Image_Pose, c = R p + t, and goes on only when z = c₃ is above 0. Its projection
  with the Shared_Camera at its full size, u = fx c₁ / z + cx and v = fy c₂ / z + cy, is in
  COLMAP's pixel convention, where the top-left pixel's centre is (0.5, 0.5), and the point goes
  on only when 0 ≤ u < W and 0 ≤ v < H. Its Prior_Size pixel is (min(⌊u w / W⌋, w − 1),
  min(⌊v h / H⌋, h − 1)), the stored pixel whose footprint holds (u, v) under Req 7.2's
  convention, where the min only guards a product that rounds up to w. The point is a Fit_Point
  when that pixel is a Geometry_Pixel. Its Target is 1 / z and its MoGe_Value 1 / MoGe's depth at
  the pixel, both float64, in the order of the image's Observed_Points.
- spread: the Spread, mean(|x − μ(x)|) with numpy's median μ, and the median.
- depth_fit (Req 5.2–5.9, 6.1): the two-pass Depth_Fit of the design's steps 1 to 8, in float64,
  with N = fit_min_points, r = fit_miss_ratio and E = fit_max_median_error:
    1. fewer than N Fit_Points: "few_points";
    2. the first fit: a Spread of the MoGe_Values of 0 is "flat", and otherwise the scale is
       Spread(t) / Spread(m) and the offset μ(t) − scale × μ(m);
    3. each point's Miss, |scale × m + offset − t|, and the points whose Miss is at most r × the
       median Miss are kept;
    4. fewer than N kept: "few_points", and a Spread of the kept MoGe_Values of 0: "flat";
    5. the second fit, on the kept points with the same formulas, and a scale of 0: "flat";
    6. over every Fit_Point, the dropped ones included, p = scale × m + offset and the
       Relative_Depth_Error |t / p − 1|, which is |1 / p − z| / z, or infinity where p isn't
       above 0;
    7. a median Relative_Depth_Error above E: "poor_fit", and otherwise "fitted";
    8. a Fitted_Frame's Metres_Per_Unit, μ(t / m) over the kept points, MoGe's depth / z.
  numpy's median sorts infinities last, so a median that lands on or beside one is infinite and
  fails step 7.
- prior_arrays (Req 5.11, 7.1): a Fitted_Frame's Prior. The Fitted_Inverse_Depth, scale ×
  (1 / MoGe's depth) + offset, is worked out in float64 at the Geometry_Pixels and rounded to
  float32. A pixel keeps its prior only where it's a Geometry_Pixel, the float32 value is finite
  and above 0, and every float32 component of MoGe's normal is finite, so every Prior_File passes
  the Train_Step's checks (Req 10.4). Every other pixel gets 0s and a false mask.
- job_metres_per_unit (Req 6.2): numpy's median of the Fitted_Frames' Metres_Per_Unit.

No number here is a setting: the thresholds come from DepthSettings, and the kinds of
FitOutcome are Fixed_Constants that report.py counts.

From splatpipe, this module imports splatpipe.train.colmap's observed_rows and
splatpipe.train.scene's world_to_camera, and colmap.py, priors.py, settings.py and gpu_api.py for
their types.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from splatpipe.depth.gpu_api import MoGeOutput
from splatpipe.depth.settings import DepthSettings
from splatpipe.train.colmap import ModelImage, SparseModel, observed_rows
from splatpipe.train.priors import Prior, PriorEntry
from splatpipe.train.scene import world_to_camera

# Fixed_Constants: what a Depth_Fit gives, a Fitted_Frame or one of the three reasons a Frame is
# left out (Req 5.2, 5.3, 5.6, 5.7, 5.9).
FITTED = "fitted"
FEW_POINTS = "few_points"
FLAT = "flat"
POOR_FIT = "poor_fit"


@dataclass(frozen=True)
class FitPoints:
    """One Frame's Fit_Points, in the order of its Observed_Points (Req 5.1)."""

    targets: np.ndarray  # (n,) float64: 1 / z, in 1 / camera solve units
    moge_values: np.ndarray  # (n,) float64: 1 / MoGe's depth, in 1 / metres


@dataclass(frozen=True)
class FitOutcome:
    """What one Frame's Depth_Fit gives. The four floats are None unless kind is FITTED."""

    kind: str  # FITTED, FEW_POINTS, FLAT or POOR_FIT
    fit_points: int  # the first fit's points
    kept_points: int  # the second fit's points, 0 when there was no second fit
    scale: float | None  # the second fit's
    offset: float | None
    median_error: float | None  # the median Relative_Depth_Error over every Fit_Point
    metres_per_unit: float | None  # the Frame's Metres_Per_Unit


def field_of_view(width: int, fx: float) -> float:
    """The Field_Of_View in degrees, from the Shared_Camera's full `width` and `fx`, both in
    pixels (Req 4.3)."""
    return math.degrees(2 * math.atan(width / (2 * fx)))


def geometry_pixels(output: MoGeOutput) -> np.ndarray:
    """The Geometry_Pixels of `output`, (h, w) bool: MoGe's mask, where its depth is finite and
    above 0 too (Req 4.7)."""
    depth = output.depth
    return np.asarray(output.mask, dtype=bool) & np.isfinite(depth) & (depth > 0)


def fit_points(model: SparseModel, image: ModelImage, output: MoGeOutput,
               width: int, height: int) -> FitPoints:
    """The Fit_Points of `image`, a Frame of `model`, against its MoGe_Output `output` at the
    Prior_Size `width` × `height` (Req 5.1, A2).

    Each Observed_Point goes on only when it's in front of the camera, its projection with the
    Shared_Camera at its full size lands inside the Frame, and its Prior_Size pixel is a
    Geometry_Pixel. The comparisons leave out a projection that overflowed.
    """
    camera = model.camera
    pose = world_to_camera(image.quaternion, image.translation)
    in_camera = model.points[observed_rows(model, image)] @ pose[:3, :3].T + pose[:3, 3]
    in_camera = in_camera[in_camera[:, 2] > 0]  # in front of the camera
    z = in_camera[:, 2]
    with np.errstate(over="ignore", invalid="ignore"):
        u = camera.fx * in_camera[:, 0] / z + camera.cx
        v = camera.fy * in_camera[:, 1] / z + camera.cy
        inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    z, u, v = z[inside], u[inside], v[inside]
    x = np.minimum(np.floor(u * width / camera.width).astype(np.int64), width - 1)
    y = np.minimum(np.floor(v * height / camera.height).astype(np.int64), height - 1)
    on_geometry = geometry_pixels(output)[y, x]
    depth = output.depth[y[on_geometry], x[on_geometry]].astype(np.float64)
    return FitPoints(targets=1.0 / z[on_geometry], moge_values=1.0 / depth)


def spread(values: np.ndarray) -> tuple[float, float]:
    """The Spread of `values`, mean(|x − μ(x)|) with numpy's median μ, and that median."""
    median = np.median(values)
    return float(np.mean(np.abs(values - median))), float(median)


def depth_fit(targets: np.ndarray, moge_values: np.ndarray,
              settings: DepthSettings) -> FitOutcome:
    """The two-pass Depth_Fit of one Frame's Fit_Points, in float64 (Req 5.2–5.9, 6.1).

    `targets` are the Fit_Points' 1 / z and `moge_values` their 1 / MoGe's depth. The median
    Relative_Depth_Error is over every Fit_Point, the ones the second fit dropped included.
    """
    t = np.asarray(targets, dtype=np.float64)
    m = np.asarray(moge_values, dtype=np.float64)
    count = int(t.size)
    if count < settings.fit_min_points:  # step 1
        return _left_out(FEW_POINTS, count)
    first = _affine(t, m)  # step 2
    if first is None:
        return _left_out(FLAT, count)
    misses = np.abs(first[0] * m + first[1] - t)  # step 3
    kept = misses <= settings.fit_miss_ratio * np.median(misses)
    kept_count = int(np.count_nonzero(kept))
    if kept_count < settings.fit_min_points:  # step 4
        return _left_out(FEW_POINTS, count)
    second = _affine(t[kept], m[kept])
    if second is None:
        return _left_out(FLAT, count)
    scale, offset = second  # step 5
    if scale == 0:
        return _left_out(FLAT, count, kept_count)
    median_error = _median_error(t, m, scale, offset)  # step 6
    if median_error > settings.fit_max_median_error:  # step 7
        return _left_out(POOR_FIT, count, kept_count)
    metres_per_unit = float(np.median(t[kept] / m[kept]))  # step 8
    return FitOutcome(FITTED, count, kept_count, scale, offset, median_error, metres_per_unit)


def prior_arrays(output: MoGeOutput, outcome: FitOutcome) -> Prior:
    """The Prior of a Fitted_Frame whose MoGe_Output is `output` and whose Depth_Fit gave
    `outcome` (Req 5.11, 7.1).

    The Fitted_Inverse_Depth is worked out in float64 at the Geometry_Pixels, inside
    np.errstate so a value that overflows gives no warning, then rounded to float32. A pixel
    keeps its prior only where it's a Geometry_Pixel, the float32 value is finite and above 0,
    and every float32 component of its normal is finite. Every other pixel gets 0s and a false
    mask.
    """
    if outcome.scale is None or outcome.offset is None:
        raise ValueError(f"prior_arrays needs a fitted Frame's outcome, not {outcome.kind!r}.")
    geometry = geometry_pixels(output)
    fitted = np.zeros(geometry.shape, dtype=np.float32)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse = 1.0 / output.depth[geometry].astype(np.float64)
        fitted[geometry] = (outcome.scale * inverse + outcome.offset).astype(np.float32)
        normal = np.asarray(output.normal, dtype=np.float32)
        keep = geometry & np.isfinite(fitted) & (fitted > 0) & np.isfinite(normal).all(axis=-1)
    return Prior(inverse_depth=np.where(keep, fitted, np.float32(0)),
                 normal=np.where(keep[..., np.newaxis], normal, np.float32(0)),
                 mask=keep)


def job_metres_per_unit(entries: Sequence[PriorEntry]) -> float:
    """The job's Metres_Per_Unit: numpy's median of the Fitted_Frames' Metres_Per_Unit, over one
    or more `entries` (Req 6.2)."""
    return float(np.median(np.array([entry.metres_per_unit for entry in entries],
                                    dtype=np.float64)))


def _affine(targets: np.ndarray, moge_values: np.ndarray) -> tuple[float, float] | None:
    """The fit of Req 5.4, (scale, offset), or None when the MoGe_Values' Spread is 0."""
    moge_spread, moge_median = spread(moge_values)
    if moge_spread == 0:
        return None
    target_spread, target_median = spread(targets)
    scale = target_spread / moge_spread
    return scale, target_median - scale * moge_median


def _median_error(targets: np.ndarray, moge_values: np.ndarray,
                  scale: float, offset: float) -> float:
    """The median Relative_Depth_Error over every Fit_Point: |t / p − 1| with p = scale × m +
    offset, or infinity where p isn't above 0 (Req 5.8)."""
    fitted = scale * moge_values + offset
    in_front = fitted > 0
    errors = np.where(in_front, np.abs(targets / np.where(in_front, fitted, 1.0) - 1.0), np.inf)
    return float(np.median(errors))


def _left_out(kind: str, fit_count: int, kept_count: int = 0) -> FitOutcome:
    """The FitOutcome of a Frame the Depth_Fit leaves out as `kind`."""
    return FitOutcome(kind, fit_count, kept_count, None, None, None, None)
