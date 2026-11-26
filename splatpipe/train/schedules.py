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
- The Loss_Table's names and weight settings (depth-priors spec, Req 13.1, 13.13): SPLAT_TERMS,
  PRIOR_TERMS and TERM_WEIGHTS, which gpu.py's tables of term functions follow.
- depth_normal_start and normal_on (depth-priors spec, Req 13.6, 13.11): when the depth_normal,
  normal_consistency and normal Loss_Terms are on.
- active_terms (depth-priors spec, Req 13.15): the names of the Loss_Terms an Iteration works
  out, in Loss_Table order. With no Prior_Set it's the splat-training rule (Req 13.13, 13.18).

This module imports only numpy, the standard library and splatpipe.train.settings, for the
TrainSettings type.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from splatpipe.train.settings import TrainSettings

MAX_SH_DEGREE = 3  # Fixed_Constant: the Splat_Layout's 45 f_rest fields hold up to degree 3

# The Loss_Table's names, in order (depth-priors spec, Req 13.1, 13.13): the splat-training
# spec's five Loss_Terms, then the five Prior_Terms that a Prior_Set adds after them.
SPLAT_TERMS = ("l1", "ssim", "opacity_reg", "scale_reg", "erank")
PRIOR_TERMS = ("depth", "normal", "flatten", "depth_normal", "normal_consistency")
# Each Loss_Term's weight setting in TrainSettings. depth has None, meaning a weight of 1,
# because its Depth_Weight is inside the term (Req 13.5).
TERM_WEIGHTS: dict[str, str | None] = {
    "l1": "l1_weight",
    "ssim": "ssim_weight",
    "opacity_reg": "opacity_reg",
    "scale_reg": "scale_reg",
    "erank": "erank_reg",
    "depth": None,
    "normal": "normal_weight",
    "flatten": "flatten_reg",
    "depth_normal": "depth_normal_weight",
    "normal_consistency": "normal_consistency_weight",
}


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


def depth_normal_start(planned: int, fraction: float) -> int:
    """The Depth_Normal_Start, the first Iteration whose loss can hold the depth_normal and
    normal_consistency Loss_Terms (depth-priors spec, Req 13.11, A4).

    It's round(fraction × planned), with fraction the setting depth_normal_start_fraction, the
    Erank_Start's rounding: 1,633 for fast, 3,500 for default and 7,000 for quality.
    """
    return round(fraction * planned)


def normal_on(iteration: int, start: int, every: int) -> bool:
    """Whether the normal Loss_Term's Iteration rule is on at `iteration` (depth-priors spec,
    Req 13.6, A4): at or after the Normal_Start `start`, and every `every`-th Iteration from it.
    """
    return iteration >= start and (iteration - start) % every == 0


def active_terms(iteration: int, settings: TrainSettings, has_priors: bool,
                 view_has_prior: bool) -> tuple[str, ...]:
    """The names of the Loss_Terms Iteration `iteration` works out, in Loss_Table order
    (depth-priors spec, Req 13.15). The Trainer counts every other row of its table as 0.

    A term whose weight is 0 is never on (Req 13.14). The five splat-training terms follow that
    spec's rule as the Trainer's _loss applied it: each is on when its weight isn't 0, and erank
    only from the Erank_Start, planned from settings.steps (Req 13.13). Only with a Prior_Set,
    `has_priors`, come the Prior_Terms: depth and normal on a Prior_View, normal on normal_on's
    Iterations, flatten on every Iteration (A6), and depth_normal and normal_consistency on a
    Prior_View from the Depth_Normal_Start. So a Training_Image with no Prior gets only the five
    splat-training terms and flatten (Req 13.12).
    """
    rules = _splat_rules(iteration, settings)
    names = SPLAT_TERMS
    if has_priors:
        rules |= _prior_rules(iteration, settings, view_has_prior)
        names = SPLAT_TERMS + PRIOR_TERMS
    return tuple(name for name in names if rules[name] and _weighted(name, settings))


def _splat_rules(iteration: int, settings: TrainSettings) -> dict[str, bool]:
    """The Iteration rule of each splat-training Loss_Term: only erank has one (Req 5.7)."""
    return {"l1": True, "ssim": True, "opacity_reg": True, "scale_reg": True,
            "erank": iteration >= erank_start(settings.steps, settings.erank_start_fraction)}


def _prior_rules(iteration: int, settings: TrainSettings,
                 view_has_prior: bool) -> dict[str, bool]:
    """The Iteration rule of each Prior_Term (depth-priors spec, Req 13.5–13.10, 13.12)."""
    from_start = iteration >= depth_normal_start(settings.steps,
                                                 settings.depth_normal_start_fraction)
    return {"depth": view_has_prior,
            "normal": view_has_prior and normal_on(iteration, settings.normal_start_step,
                                                   settings.normal_every),
            "flatten": True,
            "depth_normal": view_has_prior and from_start,
            "normal_consistency": view_has_prior and from_start}


def _weighted(name: str, settings: TrainSettings) -> bool:
    """Whether the Loss_Term `name` has a weight that isn't 0. depth's weight is 1 (Req 13.5)."""
    setting = TERM_WEIGHTS[name]
    return setting is None or getattr(settings, setting) != 0


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
