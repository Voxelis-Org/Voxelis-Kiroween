"""The Requirement 3 limits, which stop the clean step when the clean would eat the scene.

`check_source_count` refuses a trained splat that holds more than max_source_splats Gaussians.
The step runs it on the header's count, before read_splat, so the rows of a splat too big for
the job machine's memory are never read (Req 3.1).

`check_limits` runs once the filters are done and before either Step_Output is written. It
checks in this order, so a given splat and settings always give the same message (Req 3.7):

1. The Removal_Cap. The Gaussians the faint and isolated filters dropped, together, may be at
   most max_removed_fraction of the trained splat's Gaussians (Req 3.2). The broken Gaussians
   are always removed and don't count toward the cap (A4).
2. min_kept_splats. Fewer Gaussians left than that fails (Req 3.5).
3. warn_kept_fraction. Keeping less than that share of the trained splat's Gaussians gives a
   warning for the Clean_Report, and the step goes on (Req 3.6).

Past the cap the step stops. It never keeps part of the removals to get under the cap, so it
never has to rank one Gaussian against another (Req 3.4). Each failure is an ExpectedFailure,
so the job still ends with the trained splat in out/, for another try with looser settings
(Req 3.8). Both functions are pure: they work on counts and settings, and open, write and log
nothing.

From splatpipe, this module imports only splatpipe.errors, splatpipe.clean.settings and
splatpipe.clean.filters.
"""

from __future__ import annotations

from splatpipe.clean.filters import Kept
from splatpipe.clean.settings import CleanSettings
from splatpipe.errors import ExpectedFailure

# The settings in [clean] that make the faint and isolated filters drop fewer Gaussians, each
# with the way to move it, for both failures' messages (Req 3.3, 3.5).
_LOOSER_FILTERS = ("lower min_opacity, faint_min_per_cube or faint_min_paint_share, lower "
                   "faint_cube_divisions for bigger cubes, or raise isolation_ratio")


def check_source_count(count: int, s: CleanSettings) -> None:
    """Refuse a trained splat of more than max_source_splats Gaussians (Req 3.1).

    `count` is the trained splat's vertex count, from its header.
    """
    if count > s.max_source_splats:
        raise ExpectedFailure(
            f"The trained splat holds {count} Gaussians, more than the {s.max_source_splats} "
            "that max_source_splats in [clean] lets the clean step take, so that they fit in "
            "the job machine's memory. On a machine with more memory, raise max_source_splats "
            "in [clean], then run the clean step again.")


def check_limits(source: int, kept: Kept, s: CleanSettings) -> str | None:
    """The warning for the Clean_Report, or None, once the clean passes both its limits.

    `source` is the trained splat's Gaussian count, and `kept` is what run_filters kept, with
    each filter's drops. The checks come in Req 3.7's order, and each failure raises
    ExpectedFailure:

    1. The Removed_Fraction, the faint and isolated drops divided by `source` in float64, fails
       when it's above max_removed_fraction, so 7 of 10 at 0.7 passes (Req 3.2). The broken
       drops don't count toward it (A4).
    2. Fewer than min_kept_splats Gaussians left fails (Req 3.5).
    3. Less than warn_kept_fraction of `source` kept gives the warning (Req 3.6).
    """
    capped = kept.dropped_faint + kept.dropped_isolated
    removed = capped / source  # the Removed_Fraction
    left = len(kept.rows)
    if removed > s.max_removed_fraction:
        raise _over_cap(source, capped, removed, kept, s)
    if left < s.min_kept_splats:
        raise _too_few_left(source, left, kept, s)
    if left / source < s.warn_kept_fraction:
        return _kept_little(source, left, removed, s)
    return None


def _over_cap(source: int, capped: int, removed: float, kept: Kept,
              s: CleanSettings) -> ExpectedFailure:
    """The Removal_Cap's failure: the counts, the share and the cap, and what to change
    (Req 3.3)."""
    return ExpectedFailure(
        f"The faint and isolated filters would remove {capped} of the trained splat's {source} "
        f"Gaussians, {removed:.1%}, more than the cap of {s.max_removed_fraction:.1%} that "
        f"max_removed_fraction sets in [clean], so the clean step stopped. {_drops(kept)} "
        "Broken Gaussians are always removed and don't count toward the cap. To make the "
        f"filters remove less, change these settings in [clean]: {_LOOSER_FILTERS}. Or raise "
        "max_removed_fraction, so they may remove more. Then run the clean step again.")


def _too_few_left(source: int, left: int, kept: Kept, s: CleanSettings) -> ExpectedFailure:
    """The min_kept_splats failure: the counts, and what to change (Req 3.5)."""
    return ExpectedFailure(
        f"The clean would leave only {left} of the trained splat's {source} Gaussians, fewer "
        f"than the {s.min_kept_splats} that min_kept_splats asks for in [clean], so the clean "
        f"step stopped. {_drops(kept)} To make the filters remove less, change these settings "
        f"in [clean]: {_LOOSER_FILTERS}. Or, if the splat is meant to be that small, lower "
        "min_kept_splats. Then run the clean step again.")


def _kept_little(source: int, left: int, removed: float, s: CleanSettings) -> str:
    """The warning when the clean keeps less than warn_kept_fraction of the splat (Req 3.6)."""
    return (f"The clean kept only {left} of the trained splat's {source} Gaussians, "
            f"{left / source:.1%}, less than warn_kept_fraction in [clean], "
            f"{s.warn_kept_fraction:.1%}. The faint and isolated filters removed {removed:.1%} "
            "of them. Check that the clean splat still shows the whole place.")


def _drops(kept: Kept) -> str:
    """Each filter's drops, as a sentence for both failures' messages (Req 3.3, 3.5)."""
    return (f"The broken filter dropped {kept.dropped_broken}, the faint filter "
            f"{kept.dropped_faint} and the isolated filter {kept.dropped_isolated}.")
