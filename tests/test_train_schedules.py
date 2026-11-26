"""Tests for splatpipe/train/schedules.py: the Schedules that the Trainer calls.

The Schedules are pure functions of a few numbers, so each unit test passes them numbers written
here, or the real config's [train] values. The real config is read by path, because
tests/conftest.py points CONFIG_PATH at a test config. Property 3 draws its intervals and
Iterations with Hypothesis, and Property 4 its starting rates, means_lr_final values and
Planned_Iterations. The Depth_Schedule of the depth-priors spec is checked with the real config's
depth_weight_start, depth_weight_end and each Preset's steps. Its term schedules, the
Depth_Normal_Start, normal_on and active_terms, are checked with each real Preset's checked
[train] settings, with dataclasses.replace for the settings a test changes, against lists of
Loss_Terms written out here. Property 7 draws [train] values, the Planned_Iterations and an
Iteration, puts the values into the real [train] settings and checks them with read_settings, and
compares active_terms with no Prior_Set against the splat-training rule written out here.
"""

from dataclasses import replace
from itertools import islice, pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.train.schedules import (
    active_terms,
    depth_normal_start,
    depth_schedule,
    erank_start,
    image_order,
    means_lr_schedule,
    normal_on,
    refine_iterations,
    refine_stop,
    sh_degree,
)
from splatpipe.train.settings import TrainSettings, read_settings

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"


def real_train(preset: str) -> TrainSettings:
    """The real config's [train] settings for `preset`, read and checked as the step reads them."""
    return read_settings(load_config(REAL_CONFIG).step_settings("train", preset), "train", preset)


def test_the_sh_degree_goes_up_one_every_interval_and_stops_at_3() -> None:
    iterations = (0, 999, 1000, 3000, 10**6)
    assert [sh_degree(i, 1000) for i in iterations] == [0, 0, 1, 3, 3]  # Req 4.6


@st.composite
def iteration_pairs(draw: st.DrawFn) -> tuple[int, int]:
    """Two Iterations i ≤ j from 0 to 10⁷, with j drawn as i itself about half the time."""
    i = draw(st.integers(0, 10**7))
    j = draw(st.just(i) | st.integers(0, 10**7))
    return min(i, j), max(i, j)


@settings(max_examples=100, deadline=None)
@given(interval=st.integers(1, 10**6), pair=iteration_pairs())
def test_the_sh_degree_starts_at_0_never_goes_down_and_never_passes_3(
        interval: int, pair: tuple[int, int]) -> None:
    """Feature: splat-training, Property 3: The SH_Degree starts at 0, never goes down and never
    passes 3

    **Validates: Requirements 4.19, 4.6**
    """
    i, j = pair
    assert sh_degree(0, interval) == 0
    assert sh_degree(i, interval) <= sh_degree(j, interval) <= 3


@pytest.mark.parametrize(("preset", "stop", "start"), [
    ("fast", 6300, 1633),
    ("default", 13500, 3500),
    ("quality", 25000, 7000),
])
def test_the_real_config_gives_each_presets_refine_stop_and_erank_start(
        preset: str, stop: int, start: int) -> None:
    train = load_config(REAL_CONFIG).step_settings("train", preset)
    assert refine_stop(train["steps"], train["refine_stop_step"],
                       train["refine_stop_fraction"]) == stop  # Req 4.8, A8
    assert erank_start(train["steps"], train["erank_start_fraction"]) == start  # Req 5.7, A8


def test_the_means_lr_falls_from_its_start_to_1_percent_of_it_over_7000_iterations() -> None:
    start = 0.00016 * 1.1  # the real means_lr times a Scene_Extent of 1.1, as the Trainer passes
    rates = means_lr_schedule(start, 0.01, 7000)
    assert len(rates) == 7001  # one rate for each Iteration from 0 to 7,000 (Req 4.4)
    assert rates[0] == start
    assert rates[-1] == pytest.approx(start / 100, rel=1e-9, abs=0)


@settings(max_examples=100, deadline=None)
@given(start=st.floats(1e-30, 1e30),
       final=st.just(1.0) | st.floats(1e-30, 1.0),  # 1 itself about half the time
       planned=st.integers(1, 40_000))  # covers the quality Preset's 30,000
def test_the_means_lr_starts_at_its_rate_never_rises_and_ends_at_means_lr_final_of_it(
        start: float, final: float, planned: int) -> None:
    """Feature: splat-training, Property 4: The Means_LR starts at its rate, never rises and ends
    at means_lr_final of it

    **Validates: Requirements 4.20, 4.4**
    """
    rates = means_lr_schedule(start, final, planned)
    assert len(rates) == planned + 1  # one rate for each Iteration from 0 to planned
    assert rates[0] == start
    assert all(later <= earlier for earlier, later in pairwise(rates))
    # abs=0, because pytest's default absolute tolerance of 1e-12 would loosen the check for
    # every final × start below 1e-3, down to the 1e-60 these bounds allow.
    assert rates[-1] == pytest.approx(final * start, rel=1e-9, abs=0)


@pytest.mark.parametrize("preset", ["fast", "default", "quality"])
def test_the_depth_schedule_goes_from_1_to_a_tenth_halfway_and_a_hundredth_at_the_last(
        preset: str) -> None:
    train = load_config(REAL_CONFIG).step_settings("train", preset)
    planned = train["steps"]
    values = depth_schedule(train["depth_weight_start"], train["depth_weight_end"], planned)
    assert len(values) == planned + 1  # one value for each Iteration from 0 to planned (Req 11.4)
    assert values[0] == 1.0
    assert values[planned] == 0.01  # planned from the Preset's steps (Req 11.6)
    assert planned % 2 == 0  # so halfway is an Iteration
    assert values[planned // 2] == pytest.approx(0.1, rel=1e-12, abs=0)  # 1.0 × 0.01^(1/2)


def test_the_smoke_checks_30_iterations_refine_once_at_iteration_20() -> None:
    # The A11 numbers: train_steps 30, train_refine_start_step 10 and train_refine_every 10, with
    # the real refine_stop_step of 25,000 and refine_stop_fraction of 0.9 (Req 12.6).
    assert refine_iterations(30, 10, 10, refine_stop(30, 25000, 0.9)) == [20]


def test_each_pass_of_the_image_order_is_a_fresh_permutation_and_the_seed_repeats_it() -> None:
    order = list(islice(image_order(5, 42), 20))
    passes = [order[first:first + 5] for first in range(0, 20, 5)]
    assert all(sorted(one) == [0, 1, 2, 3, 4] for one in passes)  # each image once (Req 4.11)
    assert len({tuple(one) for one in passes}) > 1  # each pass drawn afresh, not one order
    assert list(islice(image_order(5, 42), 20)) == order  # the same seed, the same order (A8)


@pytest.mark.parametrize(("preset", "start"), [
    ("fast", 1633),
    ("default", 3500),
    ("quality", 7000),
])
def test_the_real_config_gives_each_presets_depth_normal_start(preset: str, start: int) -> None:
    train = real_train(preset)
    assert depth_normal_start(train.steps, train.depth_normal_start_fraction) == start  # Req 13.11


def test_the_normal_term_is_on_from_iteration_3000_and_then_every_4th_iteration() -> None:
    train = real_train("default")  # normal_start_step and normal_every are Shared_Settings
    assert (train.normal_start_step, train.normal_every) == (3000, 4)
    on = [normal_on(i, train.normal_start_step, train.normal_every)
          for i in (2999, 3000, 3003, 3004)]
    assert on == [False, True, False, True]  # Req 13.6


# The default Preset's Iterations: steps 15,000, so the Erank_Start and the Depth_Normal_Start
# are both 3,500, and the normal term is on at 3,000 and every 4th Iteration after it.
ALL_TEN = ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
           "depth", "normal", "flatten", "depth_normal", "normal_consistency")


@pytest.mark.parametrize(("iteration", "terms"), [
    (0, ("l1", "ssim", "opacity_reg", "scale_reg", "flatten")),
    (3000, ("l1", "ssim", "opacity_reg", "scale_reg", "flatten")),  # a normal Iteration
    (3499, ("l1", "ssim", "opacity_reg", "scale_reg", "flatten")),
    (3500, ("l1", "ssim", "opacity_reg", "scale_reg", "erank", "flatten")),  # the Erank_Start
    (3504, ("l1", "ssim", "opacity_reg", "scale_reg", "erank", "flatten")),  # a normal Iteration
    (14999, ("l1", "ssim", "opacity_reg", "scale_reg", "erank", "flatten")),  # the last one
])
def test_a_view_with_no_prior_gets_only_the_splat_training_terms_and_flatten(
        iteration: int, terms: tuple[str, ...]) -> None:
    # With a Prior_Set, a Training_Image with no Prior (Req 13.12, 13.15).
    assert active_terms(iteration, real_train("default"), True, False) == terms


@pytest.mark.parametrize(("iteration", "terms"), [
    (0, ("l1", "ssim", "opacity_reg", "scale_reg", "depth", "flatten")),  # depth from 0
    (2999, ("l1", "ssim", "opacity_reg", "scale_reg", "depth", "flatten")),
    (3000, ("l1", "ssim", "opacity_reg", "scale_reg", "depth", "normal", "flatten")),
    (3001, ("l1", "ssim", "opacity_reg", "scale_reg", "depth", "flatten")),
    (3499, ("l1", "ssim", "opacity_reg", "scale_reg", "depth", "flatten")),  # 499 % 4 is 3
    (3500, ALL_TEN),  # the Depth_Normal_Start, and 500 % 4 is 0
    (3501, ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
            "depth", "flatten", "depth_normal", "normal_consistency")),
    (3504, ALL_TEN),
])
def test_a_prior_view_gets_depth_from_0_normal_on_its_iterations_and_the_rest_from_their_start(
        iteration: int, terms: tuple[str, ...]) -> None:
    # Req 13.6, 13.9, 13.10, 13.15.
    assert active_terms(iteration, real_train("default"), True, True) == terms


def test_depth_normal_and_normal_consistency_start_at_the_depth_normal_start() -> None:
    # A depth_normal_start_fraction of 0.5 puts the Depth_Normal_Start at 7,500, after the
    # Erank_Start of 3,500, so the two terms follow their own start (Req 13.9, 13.10, 13.11).
    train = replace(real_train("default"), depth_normal_start_fraction=0.5)
    assert active_terms(7499, train, True, True) == (  # 4,499 % 4 is 3: no normal
        "l1", "ssim", "opacity_reg", "scale_reg", "erank", "depth", "flatten")
    assert active_terms(7500, train, True, True) == ALL_TEN  # 4,500 % 4 is 0


@pytest.mark.parametrize(("setting", "terms"), [
    ("l1_weight", ("ssim", "opacity_reg", "scale_reg", "erank",
                   "depth", "normal", "flatten", "depth_normal", "normal_consistency")),
    ("ssim_weight", ("l1", "opacity_reg", "scale_reg", "erank",
                     "depth", "normal", "flatten", "depth_normal", "normal_consistency")),
    ("opacity_reg", ("l1", "ssim", "scale_reg", "erank",
                     "depth", "normal", "flatten", "depth_normal", "normal_consistency")),
    ("scale_reg", ("l1", "ssim", "opacity_reg", "erank",
                   "depth", "normal", "flatten", "depth_normal", "normal_consistency")),
    ("erank_reg", ("l1", "ssim", "opacity_reg", "scale_reg",
                   "depth", "normal", "flatten", "depth_normal", "normal_consistency")),
    ("normal_weight", ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
                       "depth", "flatten", "depth_normal", "normal_consistency")),
    ("flatten_reg", ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
                     "depth", "normal", "depth_normal", "normal_consistency")),
    ("depth_normal_weight", ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
                             "depth", "normal", "flatten", "normal_consistency")),
    ("normal_consistency_weight", ("l1", "ssim", "opacity_reg", "scale_reg", "erank",
                                   "depth", "normal", "flatten", "depth_normal")),
])
def test_a_weight_of_0_turns_its_term_off(setting: str, terms: tuple[str, ...]) -> None:
    # Iteration 3,500 of the default Preset, where a Prior_View has all ten on (Req 13.14).
    train = replace(real_train("default"), **{setting: 0.0})
    assert active_terms(3500, train, True, True) == terms


def test_with_every_weight_0_only_depth_is_left_because_its_weight_is_1() -> None:
    weights = ("l1_weight", "ssim_weight", "opacity_reg", "scale_reg", "erank_reg",
               "normal_weight", "flatten_reg", "depth_normal_weight", "normal_consistency_weight")
    train = replace(real_train("default"), **dict.fromkeys(weights, 0.0))
    assert active_terms(3500, train, True, True) == ("depth",)  # Req 13.5, 13.14
    assert active_terms(3500, train, True, False) == ()


# Each splat-training Loss_Term, in the Loss_Table's order, with its weight setting, as the
# splat-training spec's Req 5.1 names them. Property 7's oracle reads these, not TERM_WEIGHTS.
SPLAT_TRAINING_WEIGHTS = {"l1": "l1_weight", "ssim": "ssim_weight", "opacity_reg": "opacity_reg",
                          "scale_reg": "scale_reg", "erank": "erank_reg"}


@st.composite
def splat_weights(draw: st.DrawFn) -> float:
    """A splat-training weight: 0 on about one draw in three, otherwise from 10⁻⁶ to 10."""
    if draw(st.integers(0, 2)) == 0:
        return 0.0
    return draw(st.floats(1e-6, 10))


def prior_term_settings() -> st.SearchStrategy[dict[str, int | float]]:
    """Every setting of the five Prior_Terms, each across the whole range its kind allows
    (depth-priors spec, Req 14.2), as an int or a float where TOML could give either. Each is
    finite, because the Config refuses infinity and NaN."""
    above_zero = st.floats(0, exclude_min=True, allow_infinity=False) | st.integers(1)
    zero_or_more = st.floats(0, allow_infinity=False) | st.integers(0)
    return st.fixed_dictionaries({
        "plain_gradient": above_zero,
        "depth_gradient_falloff": zero_or_more,
        "depth_weight_start": above_zero,
        "depth_weight_end": above_zero,
        "depth_weight_floor": zero_or_more,
        "depth_error_cap": above_zero,
        "normal_weight": zero_or_more,
        "normal_start_step": st.integers(0),
        "normal_every": st.integers(1),
        "flatten_reg": zero_or_more,
        "depth_normal_weight": zero_or_more,
        "normal_consistency_weight": zero_or_more,
        "depth_normal_start_fraction": st.floats(0, 1) | st.integers(0, 1),
    })


@st.composite
def no_prior_cases(draw: st.DrawFn) -> tuple[dict[str, int | float], int]:
    """[train] values for Property 7, and an Iteration below their Planned_Iterations, `steps`.

    The Iteration is the Erank_Start or one of its neighbours about half the time.
    """
    values: dict[str, int | float] = {
        setting: draw(splat_weights()) for setting in SPLAT_TRAINING_WEIGHTS.values()}
    values["erank_start_fraction"] = fraction = draw(
        st.sampled_from((0.0, 7 / 30, 1.0)) | st.floats(0, 1))  # 7/30 is the real one
    values["steps"] = planned = draw(st.integers(1, 40_000))  # covers quality's 30,000
    values |= draw(prior_term_settings())
    start = round(fraction * planned)
    near_start = [i for i in (start - 1, start, start + 1) if 0 <= i < planned]
    return values, draw(st.sampled_from(near_start) | st.integers(0, planned - 1))


def splat_training_terms(iteration: int, values: dict[str, int | float]) -> tuple[str, ...]:
    """The splat-training spec's rule (its Req 5): each of l1, ssim, opacity_reg and scale_reg
    whose weight isn't 0, then erank when its weight isn't 0 and `iteration` is at or after the
    Erank_Start, round(erank_start_fraction × Planned_Iterations) with Python's round (its A8)."""
    start = round(values["erank_start_fraction"] * values["steps"])
    on = [name for name in ("l1", "ssim", "opacity_reg", "scale_reg")
          if values[SPLAT_TRAINING_WEIGHTS[name]] != 0]
    if values[SPLAT_TRAINING_WEIGHTS["erank"]] != 0 and iteration >= start:
        on.append("erank")
    return tuple(on)


@settings(max_examples=100, deadline=None)
@given(case=no_prior_cases(), view_has_prior=st.booleans())
def test_with_no_prior_set_an_iteration_works_out_exactly_the_splat_training_terms(
        case: tuple[dict[str, int | float], int], view_has_prior: bool) -> None:
    """Feature: depth-priors, Property 7: With no Prior_Set, an Iteration works out exactly the
    splat-training terms

    **Validates: Requirements 13.18, 13.13, 13.14**
    """
    values, iteration = case
    train = load_config(REAL_CONFIG).step_settings("train", "default")
    train.update(values)  # the drawn values in place of the real ones, then checked
    checked = read_settings(train, "train", "default")
    # The oracle never holds a Prior_Term, so neither may active_terms (Req 13.13).
    assert active_terms(iteration, checked, False, view_has_prior) == (
        splat_training_terms(iteration, values))
