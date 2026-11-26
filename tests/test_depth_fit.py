"""Tests for the Depth_Step's pure maths: splatpipe/depth/fit.py (depth-priors spec, Req 4.3, 4.7,
5, 6.1 and 6.2).

Every test runs on numbers and tiny arrays it builds itself, with the real [depth] settings,
DEPTH, checked by read_settings: fit_min_points 10, fit_miss_ratio 3 and fit_max_median_error
0.15. The hand-made sets below are built for those values, and one test checks them, so a change
of them shows there.

fit_points and the Metres_Per_Unit run on tests/camera_files.py's plane scene, written into the
test's tmp_path by write_model and read back with read_model: four cameras of 16 x 12 pixels that
look at the plane z = 3 at an angle, and 30 points on it, each seen by every camera.
plane_moge_output gives each Frame's exact MoGe_Output, with the Metres_Per_Unit the test
chooses, so every Frame fits. For fit_points, the first image also names a point behind its
camera and a point on the plane outside its view, one of its points lands on a pixel where MoGe
found no geometry, and its keypoints hold the plane scene's ids of -1 and of no point. The
test's own oracle walks the image's keypoints in order and projects each point with a rotation
matrix of its own, so it shares no code with fit.py.

The Depth_Fit's unit tests use hand-made sets of Fit_Points in whole numbers, which float64
holds and adds exactly. Each set is built round the line t = m, a scale of 1 and an offset of 0.
Every point off the line has a partner on the same side of the set's median, moved by the same
amount the other way, one towards the median and one away from it. So the Targets' median and
Spread are the MoGe_Values' own, and each fit gives a scale of exactly 1 and an offset of
exactly 0. That's what lets a set keep exactly the points it means to, and put its median
Relative_Depth_Error on exactly the point it means to.

The boundary set's median error is the error of a point whose Target is 69 and whose MoGe_Value
is 60, so in exact arithmetic it's 69 / 60 - 1 = 0.15. 0.15 has no float64, and depth_fit works
the error out as |t / p - 1| and compares it with `>` against the setting's float, as the test
does: float64 gives 0.14999999999999991, which is at most the setting, so the Frame fits. Its
Target one float64 step above 69 gives 0.15000000000000013, so that Frame's fit is poor.

prior_arrays runs on MoGe_Outputs of one row of two pixels: one pixel that keeps its prior and
one that doesn't, under the same fitted scale and offset.

Property 2 draws exactly affine Fit_Points with Hypothesis, as the design's strategy says: 20 to
100 distinct MoGe_Values k / 20 for whole numbers k from 1 to 400, a scale s log-uniform from
10⁻³ to 10³, and an offset o = s × f with f from -0.9 × the smallest MoGe_Value to the largest,
so every Target, s × m + o in float64, is at least a tenth of s × the smallest MoGe_Value. With
the real settings, depth_fit must keep the Frame and give back the drawn s within a relative
error of 10⁻⁹ and the drawn o within 10⁻⁹ × max(|o|, the largest Target) (Req 5.4, 5.12). The
oracle is the drawn scale and offset, which never come from depth_fit. The MoGe_Values are
distinct because, with exact data, every Miss is rounding noise and rounding decides which
points the second fit keeps: at least half of them, so 10 or more distinct values on the 1 / 20
grid, whose Spread is at least 0.125, never 0.

Property 3 moves one to a tenth of the Targets far off, as the design's strategy says: 20 to 100
MoGe_Values a + i × d for i from 0 to n - 1, in a shuffled order, with a from 0.05 to 1 and d from
0.01 to 0.2; a scale and an offset from scales_and_offsets, with f from -0.9 a to a + (n - 1) d;
and k from 1 to ⌊n / 10⌋ distinct moved points, at least 2 choices as n is at least 20, each
Target moved up by λ × R, with λ from 6 to 7 and R the true Targets' range, s (n - 1) d up to
rounding. A moved Target only goes up, so every Target stays above 0. With the real settings,
depth_fit must keep the Frame and give back the true scale and offset within Property 2's bounds,
the offset's measured against the largest true Target (Req 5.5, 5.13). The design's argument
bounds the first fit's Misses so that every moved point's is above 3 × the median Miss, and its
note on the Spread says why every unmoved point's stays below it once k is 1 or more. k starts at
1 because with no point moved the data is exactly affine, every Miss is rounding noise, and the
3 × rule can drop unmoved points, so Property 2 covers that case. The oracle for kept_points is
the drawn points less the moved ones, counted in the test, which never asks depth_fit which
points it dropped.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.depth.fit import (
    FEW_POINTS,
    FITTED,
    FLAT,
    POOR_FIT,
    FitOutcome,
    depth_fit,
    field_of_view,
    fit_points,
    geometry_pixels,
    job_metres_per_unit,
    prior_arrays,
)
from splatpipe.depth.gpu_api import MoGeOutput
from splatpipe.depth.settings import read_settings
from splatpipe.train.colmap import SparseModel, read_model
from splatpipe.train.priors import PriorEntry

from camera_files import (PLANE_CAMERA, PLANE_IMAGES, PLANE_NO_POINT, PLANE_POINTS, PLANE_VIEWS,
                          PLANE_Z, STRAY_XY, CameraRecord, ImageRecord, PointRecord,
                          plane_moge_output, write_model)
from conftest import DEPTH

# The real [depth] settings, checked. Every [depth] setting is shared, so the Preset is any.
SETTINGS = read_settings(DEPTH, "depth", "default")

# The plane scene's Prior_Size: its camera's own 16 x 12, below [depth]'s long_edge of 512.
PRIOR_WIDTH, PRIOR_HEIGHT = PLANE_CAMERA.width, PLANE_CAMERA.height

# The two points the first image of the plane scene also names, ids 31 and 32: one on its
# optical axis behind its camera, which would project to the Frame's centre if the z > 0 check
# were missing, and one on the plane far outside its view. PLANE_GRID's point 8 lands on a
# pixel where MoGe found no geometry.
CENTRE, AIMED_AT = PLANE_VIEWS[0]
BEHIND = tuple(c - 0.5 * (a - c) for c, a in zip(CENTRE, AIMED_AT))
OUTSIDE = (4.0, 4.0, PLANE_Z)
BEHIND_ID, OUTSIDE_ID, NO_GEOMETRY_ID = 31, 32, 8


# --- The Field_Of_View and the Geometry_Pixels ---------------------------------------------------

def test_field_of_view_of_an_fx_of_half_the_width_is_90_degrees() -> None:
    """A width of 1920 pixels and an fx of 960: 2 atan(1920 / 1920) is 90° (Req 4.3)."""
    assert field_of_view(1920, 960.0) == pytest.approx(90.0, rel=1e-12)


def test_geometry_pixels_leave_out_a_false_mask_an_infinite_depth_and_a_depth_of_0() -> None:
    """Only the pixel with a true mask and a finite depth above 0 is a Geometry_Pixel (Req 4.7)."""
    output = MoGeOutput(depth=np.array([[2.0, 2.0, np.inf, 0.0]], dtype=np.float32),
                        normal=np.zeros((1, 4, 3), dtype=np.float32),
                        mask=np.array([[True, False, True, True]]))
    np.testing.assert_array_equal(geometry_pixels(output), [[True, False, False, False]])


# --- fit_points on the plane scene ----------------------------------------------------------------

def test_fit_points_keep_only_the_points_in_front_inside_the_frame_and_on_geometry(
        tmp_path: Path) -> None:
    """fit_points keeps the plane's points that are in front of the camera, inside the Frame and
    on a Geometry_Pixel, in the order of the image's keypoints, with Targets 1 / z and MoGe_Values
    1 / MoGe's depth. It leaves out the point behind the camera, the one outside the image, the
    one on a pixel with no geometry, and the keypoints of id -1 and of no point (Req 5.1)."""
    record, points = first_image_with_strays()
    model = written_model(tmp_path, (record, *PLANE_IMAGES[1:]), points)
    image = model.images[0]
    positions = {point.point_id: point.position for point in points}
    no_geometry = pixel_of(PLANE_CAMERA, record, positions[NO_GEOMETRY_ID])
    output = plane_moge_output(model, image, PRIOR_WIDTH, PRIOR_HEIGHT, 2.5,
                               no_geometry=[no_geometry])

    kept, left_out, targets, moge_values = oracle_fit_points(PLANE_CAMERA, record, positions,
                                                             output)
    assert left_out == {-1: "no point", PLANE_NO_POINT: "no point", BEHIND_ID: "behind",
                        OUTSIDE_ID: "outside", NO_GEOMETRY_ID: "no geometry"}
    assert kept == [number for number in range(1, 31) if number != NO_GEOMETRY_ID]

    found = fit_points(model, image, output, PRIOR_WIDTH, PRIOR_HEIGHT)
    assert found.targets.dtype == np.float64 and found.moge_values.dtype == np.float64
    np.testing.assert_allclose(found.targets, targets, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(found.moge_values, moge_values)


# --- depth_fit on hand-made Fit_Points -------------------------------------------------------------

def test_the_hand_made_sets_are_built_for_the_real_settings() -> None:
    """The sets below count points against fit_min_points 10, keep them with fit_miss_ratio 3 and
    put their errors round fit_max_median_error 0.15, the real values of Req 8.2."""
    assert (SETTINGS.fit_min_points, SETTINGS.fit_miss_ratio,
            SETTINGS.fit_max_median_error) == (10, 3.0, 0.15)


def test_depth_fit_leaves_out_9_points_as_too_few() -> None:
    """9 Fit_Points, one fewer than fit_min_points, exactly on a line: "few_points" (Req 5.2)."""
    outcome = depth_fit(*fit_arrays([(m, m) for m in range(1, 10)]), SETTINGS)
    assert outcome == FitOutcome(FEW_POINTS, 9, 0, None, None, None, None)


def test_depth_fit_leaves_out_a_second_fit_left_with_9_points_as_too_few() -> None:
    """11 Fit_Points, 9 on t = m and 2 moved 2 off it, (10, 8) towards the median of 6 and
    (11, 13) away. The first fit is t = m exactly, so the median Miss is 0 and the second fit
    keeps only the 9 with a Miss of 0: "few_points" (Req 5.6)."""
    pairs = [(m, m) for m in range(1, 10)] + [(10, 8), (11, 13)]
    outcome = depth_fit(*fit_arrays(pairs), SETTINGS)
    assert outcome == FitOutcome(FEW_POINTS, 11, 0, None, None, None, None)


def test_depth_fit_leaves_out_equal_moge_values_as_flat() -> None:
    """10 Fit_Points whose MoGe_Values are all 0.5: their Spread is 0, so "flat" (Req 5.3)."""
    outcome = depth_fit(*fit_arrays([(0.5, t) for t in range(1, 11)]), SETTINGS)
    assert outcome == FitOutcome(FLAT, 10, 0, None, None, None, None)


def test_depth_fit_leaves_out_a_second_fit_on_equal_moge_values_as_flat() -> None:
    """20 Fit_Points: 18 at (1, 1), (2, 5) and (3, 3). The first fit is t = 2m - 1 exactly, a
    Spread of 6 / 20 over one of 3 / 20, which the 18 lie on and the other two are 2 off, so the
    median Miss is 0 and the second fit keeps only the 18, whose MoGe_Values are all 1: "flat"
    (Req 5.6)."""
    outcome = depth_fit(*fit_arrays([(1, 1)] * 18 + [(2, 5), (3, 3)]), SETTINGS)
    assert outcome == FitOutcome(FLAT, 20, 0, None, None, None, None)


def test_depth_fit_leaves_out_a_second_fit_scale_of_0_as_flat() -> None:
    """10 Fit_Points whose Targets are all 2 and whose MoGe_Values are 1 to 10: every Miss is 0,
    so the second fit keeps all 10, and its scale is 0: "flat" (Req 5.7)."""
    outcome = depth_fit(*fit_arrays([(m, 2) for m in range(1, 11)]), SETTINGS)
    assert outcome == FitOutcome(FLAT, 10, 10, None, None, None, None)


# 21 hand-made Fit_Points round the line t = m, as (MoGe_Value, Target) pairs: 9 on it, 4 near it
# and kept, and 8 far off it and dropped. The near points sit below both fits' medians, the far
# ones above the first fit's, each pair moved the same amount, one towards the median and one away,
# so both fits are t = m exactly: the first's medians are 23 and the second's 19.
ON_LINE = tuple((m, m) for m in range(17, 26))  # a Miss of 0 and an error of 0
NEAR = ((4, 2), (6, 8), (8, 6), (16, 18))  # a Miss of 2, the median Miss: errors 1/2 to 1/8
FAR_TOWARDS = ((34, 25), (38, 29), (42, 33), (46, 37))  # a Miss of 9, above 3 x 2: dropped
# The other 4 far points, each moved 9 away from the median, (m, m + 9): errors 9 / m.
FAR_AWAY_POOR = ((36, 45), (40, 49), (44, 53), (48, 57))  # errors from 1/4 to 0.1875
FAR_AWAY_BOUNDARY = ((50, 59), (54, 63), (58, 67), (60, 69))  # the last one's error is 0.15


def test_depth_fit_gives_a_poor_fit_when_the_dropped_points_push_the_median_error_over() -> None:
    """The second fit keeps the 13 points on and near the line and drops the 8 far ones. Over
    the 13 alone the median Relative_Depth_Error is 0, but it's over every Fit_Point, the dropped
    ones included, so it's the 11th smallest of 21, 9 / 48 = 0.1875: "poor_fit" (Req 5.8, 5.9)."""
    pairs = ON_LINE + NEAR + FAR_TOWARDS + FAR_AWAY_POOR
    assert median_error_on_the_line(ON_LINE + NEAR) == 0.0
    assert median_error_on_the_line(pairs) == pytest.approx(0.1875, rel=1e-12)
    outcome = depth_fit(*fit_arrays(pairs), SETTINGS)
    assert outcome == FitOutcome(POOR_FIT, 21, 13, None, None, None, None)


def test_depth_fit_keeps_a_median_error_of_exactly_0_15() -> None:
    """The median Relative_Depth_Error is the error of (60, 69), 69 / 60 - 1, which is 0.15 in
    exact arithmetic and at most the setting in float64: "fitted", with the scale 1, the offset 0
    and a Metres_Per_Unit of 1, the median of the 13 kept points' t / m (Req 5.9, 6.1)."""
    pairs = ON_LINE + NEAR + FAR_TOWARDS + FAR_AWAY_BOUNDARY
    error = abs(69.0 / 60.0 - 1.0)  # as depth_fit works it out: |t / p - 1|, with p = 1 × 60 + 0
    assert not error > SETTINGS.fit_max_median_error
    assert median_error_on_the_line(pairs) == error
    outcome = depth_fit(*fit_arrays(pairs), SETTINGS)
    assert outcome == FitOutcome(FITTED, 21, 13, 1.0, 0.0, error, 1.0)


def test_depth_fit_gives_a_poor_fit_for_a_median_error_just_above_0_15() -> None:
    """The boundary set with the Target 69 one float64 step higher: its error is above 0.15 in
    exact arithmetic and in float64. It's still dropped from the second fit, which is still
    t = m exactly, and it's still the median: "poor_fit" (Req 5.9)."""
    *others, (moge_value, target) = ON_LINE + NEAR + FAR_TOWARDS + FAR_AWAY_BOUNDARY
    pairs = (*others, (moge_value, float(np.nextafter(float(target), math.inf))))
    assert median_error_on_the_line(pairs) > SETTINGS.fit_max_median_error
    outcome = depth_fit(*fit_arrays(pairs), SETTINGS)
    assert outcome == FitOutcome(POOR_FIT, 21, 13, None, None, None, None)


# --- prior_arrays ------------------------------------------------------------------------------------

FACING = (0.0, 0.6, -0.8)  # a unit normal facing the camera

# Each case: what's wrong with the first pixel, the fit's scale and offset, the first pixel's
# depth and normal, and the second pixel's depth, at which the same fit gives a value float32
# holds. MoGe's depths are float32, so 1e-30 and 1e10 are a float32's.
DROPPED_PIXELS = [
    ("a negative fitted value", 1.0, -1.0, 2.0, FACING, 0.25),  # 1 / 2 - 1 = -0.5
    ("a fitted value that rounds to 0 in float32", 1e-50, 0.0, 1.0, FACING, 1e-30),  # 1e-50
    ("a fitted value that overflows float32", 1e39, 0.0, 1.0, FACING, 1e10),  # 1e39
    ("a normal that isn't finite", 1.0, 0.0, 2.0, (math.nan, 0.6, -0.8), 2.0),
]


@pytest.mark.parametrize(("why", "scale", "offset", "depth", "normal", "good_depth"),
                         DROPPED_PIXELS, ids=[case[0] for case in DROPPED_PIXELS])
def test_prior_arrays_give_a_pixel_0s_and_a_false_mask(
        why: str, scale: float, offset: float, depth: float, normal: tuple[float, float, float],
        good_depth: float) -> None:
    """A Geometry_Pixel with a fitted value that's negative, that rounds to 0 or overflows in
    float32, or with a normal that isn't finite gets 0s and a false mask, while the pixel beside
    it keeps scale × 1 / depth + offset, rounded to float32, and its normal (Req 5.11)."""
    output = MoGeOutput(depth=np.array([[depth, good_depth]], dtype=np.float32),
                        normal=np.array([[normal, FACING]], dtype=np.float32),
                        mask=np.array([[True, True]]))
    prior = prior_arrays(output, FitOutcome(FITTED, 10, 10, scale, offset, 0.0, 1.0))

    assert prior.inverse_depth.dtype == np.float32 and prior.normal.dtype == np.float32
    assert prior.mask.dtype == np.bool_
    np.testing.assert_array_equal(prior.mask, [[False, True]])
    kept = np.float32(scale * (1.0 / float(np.float32(good_depth))) + offset)
    np.testing.assert_array_equal(prior.inverse_depth, np.array([[0.0, kept]], dtype=np.float32))
    np.testing.assert_array_equal(prior.normal,
                                  np.array([[(0.0, 0.0, 0.0), FACING]], dtype=np.float32))


# --- The Metres_Per_Unit -----------------------------------------------------------------------------

def test_the_plane_scene_gives_the_chosen_metres_per_unit_per_frame_and_for_the_job(
        tmp_path: Path) -> None:
    """Each Frame of the plane scene, with MoGe's depths its own Metres_Per_Unit times the camera
    solve's z, fits with that Metres_Per_Unit, and the job's is their median: 3, the mean of 2.5
    and 3.5, not the mean of all four (Req 6.1, 6.2). MoGe's depths are float32, so the second
    fit keeps at least fit_min_points of the 30 points rather than all of them."""
    chosen = (2.0, 2.5, 3.5, 10.0)
    model = written_model(tmp_path, PLANE_IMAGES, PLANE_POINTS)
    entries = []
    for image, metres_per_unit in zip(model.images, chosen):
        output = plane_moge_output(model, image, PRIOR_WIDTH, PRIOR_HEIGHT, metres_per_unit)
        found = fit_points(model, image, output, PRIOR_WIDTH, PRIOR_HEIGHT)
        outcome = depth_fit(found.targets, found.moge_values, SETTINGS)
        assert outcome.kind == FITTED and outcome.fit_points == 30
        assert outcome.kept_points >= SETTINGS.fit_min_points
        assert outcome.metres_per_unit == pytest.approx(metres_per_unit, rel=1e-6)
        entries.append(PriorEntry(
            name=image.name, quaternion=tuple(image.quaternion.tolist()),
            translation=tuple(image.translation.tolist()), scale=outcome.scale,
            offset=outcome.offset, fit_points=outcome.fit_points,
            kept_points=outcome.kept_points, median_error=outcome.median_error,
            metres_per_unit=outcome.metres_per_unit))
    assert job_metres_per_unit(entries) == pytest.approx(3.0, rel=1e-6)


# --- Property 2: exactly affine data is recovered ------------------------------------------------

# Req 5.12's bound: the scale's relative error, and the offset's error as a share of
# max(|offset|, the largest Target).
RECOVERY_BOUND = 1e-9

# The number of Fit_Points, from 2 × fit_min_points, so the second fit keeps at least
# fit_min_points, to 100.
FIT_POINT_COUNTS = st.integers(min_value=20, max_value=100)

# The scale, log-uniform from 10⁻³ to 10³: 10 to the power of an exponent from -3 to 3.
SCALES = st.floats(min_value=-3.0, max_value=3.0).map(lambda exponent: 10.0 ** exponent)


@st.composite
def scales_and_offsets(draw: st.DrawFn, smallest: float, largest: float) -> tuple[float, float]:
    """A scale s from SCALES and an offset o = s × f, with f from -0.9 × `smallest` to
    `largest`, the smallest and largest MoGe_Values. Every Target s × m + o is then at least
    0.1 × s × `smallest`, so above 0, and |o| is at most s × `largest`."""
    scale = draw(SCALES)
    return scale, scale * draw(st.floats(min_value=-0.9 * smallest, max_value=largest))


@st.composite
def exactly_affine_sets(draw: st.DrawFn) -> tuple[np.ndarray, float, float]:
    """The MoGe_Values, (n,) float64, of 20 to 100 distinct whole numbers k from 1 to 400, each
    k / 20, so from 0.05 to 20, with a scale and an offset from scales_and_offsets."""
    count = draw(FIT_POINT_COUNTS)
    whole = draw(st.lists(st.integers(min_value=1, max_value=400), min_size=count,
                          max_size=count, unique=True))
    moge_values = np.array(whole, dtype=np.float64) / 20.0
    scale, offset = draw(scales_and_offsets(float(moge_values.min()), float(moge_values.max())))
    return moge_values, scale, offset


@settings(max_examples=100, deadline=None)
@given(drawn=exactly_affine_sets())
def test_depth_fit_recovers_the_scale_and_offset_of_exactly_affine_data(
        drawn: tuple[np.ndarray, float, float]) -> None:
    """Feature: depth-priors, Property 2: Exactly affine data is recovered

    Every Target is exactly the drawn scale × its MoGe_Value + the drawn offset, in float64, so
    depth_fit with the real settings keeps the Frame and gives back that scale and offset within
    Req 5.12's bounds.

    **Validates: Requirements 5.12, 5.4**
    """
    moge_values, scale, offset = drawn
    targets = scale * moge_values + offset
    assert (targets > 0).all()  # the strategy's bounds keep every Target above 0
    outcome = depth_fit(targets, moge_values, SETTINGS)
    assert_recovered(outcome, scale, offset, float(targets.max()))


# --- Property 3: up to a tenth of points moved far off are dropped -------------------------------

# The MoGe_Values' first value a and their step d: each MoGe_Value is a + i × d.
FIRST_VALUES = st.floats(min_value=0.05, max_value=1.0)
STEPS = st.floats(min_value=0.01, max_value=0.2)

# λ: a moved Target goes up by λ × the true Targets' range.
MOVE_FACTORS = st.floats(min_value=6.0, max_value=7.0)


@dataclass(frozen=True)
class MovedSet:
    """One drawn set of Fit_Points with some Targets moved far off."""

    moge_values: np.ndarray  # (n,) float64: a + i × d, in a shuffled order
    targets: np.ndarray  # (n,) float64: the true Targets, with the moved ones moved up
    scale: float  # the true scale
    offset: float  # the true offset
    largest_true_target: float
    moved: tuple[int, ...]  # the indices of the moved Targets, distinct


@st.composite
def sets_with_moved_points(draw: st.DrawFn) -> MovedSet:
    """20 to 100 MoGe_Values a + i × d in a shuffled order, a scale and an offset from
    scales_and_offsets, and 1 to ⌊n / 10⌋ distinct Targets, each moved up by λ × R, with λ
    from MOVE_FACTORS and R the true Targets' range."""
    count = draw(FIT_POINT_COUNTS)
    first, step = draw(FIRST_VALUES), draw(STEPS)
    order = draw(st.permutations(range(count)))
    moge_values = first + np.array(order, dtype=np.float64) * step
    scale, offset = draw(scales_and_offsets(float(moge_values.min()), float(moge_values.max())))
    true_targets = scale * moge_values + offset
    moved = draw(st.lists(st.integers(min_value=0, max_value=count - 1), min_size=1,
                          max_size=count // 10, unique=True))
    factors = draw(st.lists(MOVE_FACTORS, min_size=len(moved), max_size=len(moved)))
    true_range = float(true_targets.max() - true_targets.min())
    targets = true_targets.copy()
    targets[moved] += np.array(factors, dtype=np.float64) * true_range
    return MovedSet(moge_values, targets, scale, offset, float(true_targets.max()), tuple(moved))


@settings(max_examples=100, deadline=None)
@given(drawn=sets_with_moved_points())
def test_depth_fit_drops_up_to_a_tenth_of_points_moved_far_off_and_recovers_the_fit(
        drawn: MovedSet) -> None:
    """Feature: depth-priors, Property 3: Up to a tenth of points moved far off are dropped, and
    the true fit is recovered

    With one to a tenth of the Targets moved up by 6 to 7 times the true Targets' range, depth_fit
    with the real settings drops exactly the moved points from its second fit, keeps the Frame
    and gives back the true scale and offset within Req 5.12's bounds.

    **Validates: Requirements 5.13, 5.5**
    """
    assert (drawn.targets > 0).all()  # true Targets are above 0 and moved ones only go up
    outcome = depth_fit(drawn.targets, drawn.moge_values, SETTINGS)
    assert_recovered(outcome, drawn.scale, drawn.offset, drawn.largest_true_target)
    kept = [index for index in range(drawn.targets.size) if index not in drawn.moved]
    assert outcome.kept_points == len(kept), (outcome.kept_points, len(kept), len(drawn.moved))


# --- Helpers -------------------------------------------------------------------------------------------

def fit_arrays(pairs: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """depth_fit's arguments from (MoGe_Value, Target) `pairs`: the Targets, then the
    MoGe_Values, each (n,) float64."""
    moge_values, targets = zip(*pairs)
    return np.array(targets, dtype=np.float64), np.array(moge_values, dtype=np.float64)


def assert_recovered(outcome: FitOutcome, scale: float, offset: float,
                     largest_target: float) -> None:
    """`outcome` is a Fitted_Frame whose scale is within a relative error of RECOVERY_BOUND of
    the true `scale`, and whose offset is within RECOVERY_BOUND × max(|`offset`|,
    `largest_target`) of the true `offset` (Req 5.12)."""
    assert outcome.kind == FITTED, outcome
    assert outcome.scale is not None and outcome.offset is not None
    assert abs(outcome.scale - scale) <= RECOVERY_BOUND * scale, (outcome.scale, scale)
    assert abs(outcome.offset - offset) <= RECOVERY_BOUND * max(abs(offset), largest_target), (
        outcome.offset, offset)


def median_error_on_the_line(pairs: Sequence[tuple[float, float]]) -> float:
    """numpy's median of the Relative_Depth_Errors |t / m - 1| of (MoGe_Value, Target) `pairs`
    against the line t = m, the fit every hand-made set comes out at."""
    targets, moge_values = fit_arrays(pairs)
    return float(np.median(np.abs(targets / moge_values - 1.0)))


def first_image_with_strays() -> tuple[ImageRecord, tuple[PointRecord, ...]]:
    """The plane scene's first image with keypoints for BEHIND and OUTSIDE added at its end, and
    the plane scene's points with those two added, seen by the first image alone."""
    record = PLANE_IMAGES[0]
    count = len(record.keypoints)
    extra = ((BEHIND_ID, BEHIND), (OUTSIDE_ID, OUTSIDE))
    keypoints = record.keypoints + tuple((*STRAY_XY, point_id) for point_id, _ in extra)
    points = PLANE_POINTS + tuple(
        PointRecord(point_id=point_id, position=position, colour=(255, 255, 255),
                    track=((record.image_id, count + place),))
        for place, (point_id, position) in enumerate(extra))
    return replace(record, keypoints=keypoints), points


def written_model(tmp_path: Path, images: Sequence[ImageRecord],
                  points: Sequence[PointRecord]) -> SparseModel:
    """The plane scene's model with `images` and `points`, written by write_model and read back
    with read_model."""
    write_model(tmp_path / "sparse", PLANE_CAMERA, images, points)
    return read_model(tmp_path / "sparse")


def oracle_fit_points(camera: CameraRecord, record: ImageRecord,
                      positions: dict[int, tuple[float, float, float]],
                      output: MoGeOutput) -> tuple[list[int], dict[int, str], list[float],
                                                   list[float]]:
    """The Fit_Points of `record`, worked out here: the ids kept in keypoint order, each left-out
    id with why ("no point", "behind", "outside" or "no geometry"), and the kept points' Targets
    and MoGe_Values. An id already seen is skipped, as an Observed_Point is distinct."""
    height, width = output.mask.shape
    kept: list[int] = []
    left_out: dict[int, str] = {}
    targets: list[float] = []
    moge_values: list[float] = []
    for *_, point_id in record.keypoints:
        if point_id in kept or point_id in left_out:
            continue
        if point_id not in positions:
            left_out[point_id] = "no point"
            continue
        x, y, z = camera_space(record, positions[point_id])
        u, v = camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy
        if z <= 0:
            left_out[point_id] = "behind"
        elif not (0 <= u < camera.width and 0 <= v < camera.height):
            left_out[point_id] = "outside"
        else:
            column = min(math.floor(u * width / camera.width), width - 1)
            row = min(math.floor(v * height / camera.height), height - 1)
            depth = float(output.depth[row, column])
            if not (output.mask[row, column] and math.isfinite(depth) and depth > 0):
                left_out[point_id] = "no geometry"
            else:
                kept.append(point_id)
                targets.append(1.0 / z)
                moge_values.append(1.0 / depth)
    return kept, left_out, targets, moge_values


def pixel_of(camera: CameraRecord, record: ImageRecord,
             position: tuple[float, float, float]) -> tuple[int, int]:
    """The (column, row) of the Prior_Size pixel, the camera's own size here, that `position`
    lands on in `record`'s image."""
    x, y, z = camera_space(record, position)
    return (math.floor(camera.fx * x / z + camera.cx), math.floor(camera.fy * y / z + camera.cy))


def camera_space(record: ImageRecord, position: tuple[float, float, float]) -> np.ndarray:
    """`position` in the camera frame of `record`, R p + t, with R the rotation matrix of its
    quaternion (w, x, y, z) once scaled to length 1."""
    w, x, y, z = np.asarray(record.quaternion, dtype=np.float64) / np.linalg.norm(
        record.quaternion)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rotation @ np.asarray(position, dtype=np.float64) + np.asarray(record.translation,
                                                                          dtype=np.float64)
