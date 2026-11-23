"""The Levelling's settings (Req 9).

LevelSettings holds the four settings of [cameras] that the Levelling uses. settings.py reads and
checks them, and keeps them in CamerasSettings.level.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelSettings:
    """The Levelling's four settings from [cameras], each as Config gave it."""

    up_agreement: float  # the shortest Mean_Up at which the cameras agree on which way is up
    heading_min_degrees: float  # a camera within this of straight up or down has no Heading
    axis_point_min_eigen_share: float  # the smallest eigenvalue share of a usable Axis_Point
    axis_point_min_front_share: float  # the share of cameras a usable Axis_Point is in front of
