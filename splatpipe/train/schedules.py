"""The Schedules: pure functions of the Iteration that the Trainer in gpu.py calls.

They import no torch, so the Test_Suite checks every schedule on a laptop (Req 4.18). Iterations
count from 0, and `planned` is the Preset's `steps`, the Planned_Iterations.

- sh_degree (Req 4.6): min(i // sh_degree_interval, 3), so the SH_Degree goes up by one every
  sh_degree_interval Iterations and stops at MAX_SH_DEGREE.
- means_lr_schedule (Req 4.4): the Means_LR of every Iteration from 0 to planned, falling
  exponentially from its start to means_lr_final of it. There's no torch scheduler.
- refine_stop (Req 4.8) and erank_start (Req 5.7): Iterations worked out from fractions of the
  Planned_Iterations, rounded with Python's round, which takes a tie to the even neighbour (A8).
- refine_iterations: the Iterations at which gsplat 1.5.3's MCMCStrategy relocates and adds
  splats, so the Train_Smoke_Check can count its refines.
- image_order (Req 4.11): one Training_Image index per Iteration, each pass a fresh permutation
  from one generator seeded with `seed` (A8).
- depth_schedule (depth-priors spec, Req 11.4, 11.6, A4): the Depth_Schedule of every Iteration
  from 0 to planned, going exponentially from depth_weight_start to depth_weight_end.

This module imports only numpy and the standard library.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

MAX_SH_DEGREE = 3  # Fixed_Constant: the Splat_Layout's 45 f_rest fields hold up to degree 3


def sh_degree(iteration: int, interval: int) -> int:
    """The SH_Degree that Iteration `iteration` renders with: min(iteration // interval, 3)."""
    return min(iteration // interval, MAX_SH_DEGREE)


def means_lr_schedule(start: float, final: float, planned: int) -> tuple[float, ...]:
    """The Means_LR of each Iteration from 0 to `planned`: planned + 1 rates (Req 4.4).

    Rate i is start × final^(i / planned), the rate Iteration i trains at, and rate `planned` is
    the rate once the last Planned_Iteration has ended. The two ends are set outright, as start
    and start × final, so they don't lean on pow(x, 1.0) giving exactly x, which C doesn't
    promise. Then a running minimum keeps any rate from rising above the one before it, which a
    platform's pow can do by one ULP, because it isn't always correctly rounded. Elsewhere the
    minimum changes nothing, and the last rate stays within one rounding of start × final. The
    Trainer passes start = means_lr × Scene_Extent and final = means_lr_final.
    """
    between = (start * final ** (i / planned) for i in range(1, planned))
    return tuple(itertools.accumulate((start, *between, start * final), min))


def depth_schedule(start: float, end: float, planned: int) -> tuple[float, ...]:
    """The Depth_Schedule of each Iteration from 0 to `planned`: planned + 1 values
    (depth-priors spec, Req 11.4, A4).

    Value i is start × (end / start)^(i / planned). The two ends are set outright, as start and
    end, so they don't lean on pow giving exactly 1 or exactly end / start. Unlike
    means_lr_schedule there's no running minimum, because depth_weight_end may be above
    depth_weight_start. The Trainer passes depth_weight_start, depth_weight_end and the
    Planned_Iterations, whether or not it stops early (Req 11.6).
    """
    ratio = end / start
    between = (start * ratio ** (i / planned) for i in range(1, planned))
    return (start, *between, end)


def refine_stop(planned: int, refine_stop_step: int, refine_stop_fraction: float) -> int:
    """The Refine_Stop: min(refine_stop_step, round(refine_stop_fraction × planned)) (Req 4.8)."""
    return min(refine_stop_step, round(refine_stop_fraction * planned))


def erank_start(planned: int, fraction: float) -> int:
    """The Erank_Start, the first Iteration whose loss holds the erank Loss_Term (Req 5.7).

    It's round(fraction × planned), with fraction the setting erank_start_fraction.
    """
    return round(fraction * planned)


def refine_iterations(planned: int, refine_start: int, refine_every: int,
                      stop: int) -> list[int]:
    """The Iterations below `planned` at which MCMC relocates and adds splats, in order.

    They're each i with refine_start < i < stop and i % refine_every == 0, the condition gsplat
    1.5.3's MCMCStrategy checks in step_post_backward, with `stop` the Refine_Stop. The strategy
    applies the condition itself, so the Trainer doesn't call this.
    """
    return [i for i in range(planned) if refine_start < i < stop and i % refine_every == 0]


def image_order(count: int, seed: int) -> Iterator[int]:
    """The index of each Iteration's Training_Image, pass after pass, without end (Req 4.11).

    One generator, np.random.default_rng(seed), is made once, and each pass is its next
    permutation(count), as gsplat's example reshuffles its loader for each pass. So every pass
    visits each of the `count` Training_Images once, and the same seed gives the same order. The
    Trainer takes one index per Iteration. `count` is at least 1, because the Model_Reader
    refuses a model with no image.
    """
    generator = np.random.default_rng(seed)
    while True:
        yield from generator.permutation(count).tolist()
