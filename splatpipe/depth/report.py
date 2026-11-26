"""What the Depth_Step reports: the Frames it fitted and left out, its running timings, its
warning and its no-fit failure (depth-priors spec, Req 7.7, 7.9, 9.1 and 9.2).

- Tally counts the Frames by what their Depth_Fit gave. Its fields but `frames` are named after
  fit.py's kinds, "fitted", "few_points", "flat" and "poor_fit", and the step adds them to the
  report as fitted_images, few_points_images, flat_images and poor_fit_images (Req 9.1).
- Timings keeps the running sum of each per-frame timing, infer, fit and write, because
  StepContext.add_timing replaces a timing. Each add gives add_timing the new sum, so a failed
  report holds the sums up to the last block that ended (Req 9.2, 9.3).
- left_out_warning is the Req 7.9 warning, for a run that left out some Frames but fitted at
  least one.
- no_fit_failure is the Req 7.7 failure, for a run that fitted no Frame (D3).

This module reads no file and holds no setting: the two thresholds the failure names come from
DepthSettings. From splatpipe, it imports only splatpipe.errors, and splatpipe.step and
splatpipe.depth.settings for their types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from splatpipe.errors import ExpectedFailure

if TYPE_CHECKING:
    from splatpipe.depth.settings import DepthSettings
    from splatpipe.step import StepContext


@dataclass
class Tally:
    """How many Frames the step has gone through, and how many of them each kind of Depth_Fit
    gave. Each field but `frames` is named after a kind of fit.py's FitOutcome."""

    frames: int = 0
    fitted: int = 0
    few_points: int = 0
    flat: int = 0
    poor_fit: int = 0

    def add(self, kind: str) -> None:
        """Count one more Frame, whose Depth_Fit gave `kind`: fit.py's FITTED, FEW_POINTS, FLAT
        or POOR_FIT. Any other kind is a bug, and raises ValueError."""
        if kind not in _KINDS:
            raise ValueError(f"{kind!r} isn't a kind of depth fit. The kinds are "
                             f"{', '.join(_KINDS)}.")
        self.frames += 1
        setattr(self, kind, getattr(self, kind) + 1)


# The kinds of Depth_Fit that Tally counts: its fields but frames.
_KINDS = tuple(item.name for item in fields(Tally) if item.name != "frames")


@dataclass
class Timings:
    """The running sums, in seconds, of the timings the step adds once per Frame (Req 9.2)."""

    sums: dict[str, float] = field(default_factory=dict)

    def add(self, step: StepContext, name: str, seconds: float) -> None:
        """Add `seconds` to the timing `name`'s sum, and give the new sum to step.add_timing,
        which replaces the timing's earlier value."""
        total = self.sums.get(name, 0.0) + seconds
        step.add_timing(name, total)
        self.sums[name] = total


def left_out_warning(tally: Tally) -> str:
    """The Req 7.9 warning: how many Frames were fitted, and how many were left out for each
    reason."""
    return (f"Depth priors were made for {tally.fitted} of {tally.frames} frames. Of the others, "
            f"{tally.few_points} had too few of the camera solve's points, {tally.flat} were flat "
            f"and {tally.poor_fit} fitted poorly, so they train without a depth prior.")


def no_fit_failure(tally: Tally, settings: DepthSettings) -> ExpectedFailure:
    """The Req 7.7 failure for a run that fitted no Frame: the Frames, how many were left out for
    each reason, and the fit_min_points and fit_max_median_error of [depth] (D3)."""
    return ExpectedFailure(
        f"None of the {tally.frames} frames fitted the camera solve's points, so no depth priors "
        f"were made: {tally.few_points} had fewer than {settings.fit_min_points} of the points, "
        f"{tally.flat} were flat, and {tally.poor_fit} fitted poorly, with a median depth error "
        f"above {settings.fit_max_median_error}. The job goes on and trains without depth priors.")
