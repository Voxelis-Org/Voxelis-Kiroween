"""Tests for splatpipe/train/schedules.py: the Schedules that the Trainer calls.

The Schedules are pure functions of a few numbers, so each unit test passes them numbers written
here, or the real config's [train] values. The real config is read by path, because
tests/conftest.py points CONFIG_PATH at a test config. Property 3 draws its intervals and
Iterations with Hypothesis, and Property 4 its starting rates, means_lr_final values and
Planned_Iterations. The Depth_Schedule of the depth-priors spec is checked with the real config's
depth_weight_start, depth_weight_end and each Preset's steps.
"""

from itertools import islice, pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.train.schedules import (
    depth_schedule,
    erank_start,
    image_order,
    means_lr_schedule,
    refine_iterations,
    refine_stop,
    sh_degree,
)

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"


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
