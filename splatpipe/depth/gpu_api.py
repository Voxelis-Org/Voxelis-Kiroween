"""The Depth_GPU_Module's interface as the CPU sees it.

The Depth_GPU_Module, gpu.py, is the only file of the Depth_Step that imports torch, MoGe and
huggingface_hub. Everything it gives back to the CPU side has the CPU modules' own types, so the
fit and the Prior_Writer work on numpy arrays and the Test_Suite builds them on a laptop.

- MoGeOutput (Req 4.6): one Frame's MoGe_Output at the Prior_Size, as numpy arrays: depth (h, w)
  float32, camera-space z in metres; normal (h, w, 3) float32, unit normals in the camera frame
  that face the camera; and mask (h, w) bool, true where MoGe found geometry. With MoGe's
  apply_mask at its default, depth is +inf and normal 0 where mask is false (moge-depth skill).

The rest of the interface, the hook that imports gpu.py, the DepthGpu protocol, its exceptions
and the checks that turn them into Expected_Failures, isn't in this module yet.

This module imports only the standard library and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MoGeOutput:
    """One Frame's MoGe_Output at the Prior_Size h × w (Req 4.6)."""

    depth: np.ndarray  # (h, w) float32, camera-space z in metres, +inf where mask is false
    normal: np.ndarray  # (h, w, 3) float32, unit, camera frame, facing the camera; 0 off the mask
    mask: np.ndarray  # (h, w) bool, true where MoGe found geometry
