"""A Prior_View's inputs to the loss, worked out per training pixel at the Training_Size with pure
functions (depth-priors spec, Req 10.6, 10.7, 10.9 and 11).

- brightness (Req 11.1, A12): OpenCV's grey value of the 8-bit RGB Training_Image, / 255, in
  float64.
- brightness_gradient (Req 11.2, A12): the length of the central-difference gradient,
  (I(x + 1) − I(x − 1)) / 2 across and (I(y + 1) − I(y − 1)) / 2 down, with the border pixels
  repeated outward, as float32. Every later test reads this float32 array, on the CPU and on the
  GPU alike.
- plain_pixels (Req 11.3): the Plain_Pixels, whose gradient is below plain_gradient.
- depth_weight (Req 11.5, D2): rate × exp(−depth_gradient_falloff × gradient), raised to
  depth_weight_floor on a Plain_Pixel whose product is below it.
- upscale_taps and prior_taps (Req 10.6): the scale-up from the Prior_Size to the Training_Size,
  one axis at a time, worked out in integers with pixel centres aligned. Training pixel X of T,
  centred at (X + 0.5) / T of the edge, sits at stored coordinate (X + 0.5) · S / T − 0.5 of S
  stored pixels, as COLMAP and MoGe place pixel centres. The nearest stored pixel is the one
  under the training pixel's centre, (2X + 1) · S // (2T), as scene.keep_mask takes it.
- upscale (Req 10.6): bilinear sampling of an (h, w) or (h, w, 3) Prior array at the Training_Size.
  The Trainer sets each scaled-up normal to unit length itself.
- prior_pixel_mask (Req 10.7, A13): the Prior_Pixels of a view: not Masked_Pixels, a true mask
  at the nearest stored pixel, and a true mask at every stored pixel the bilinear sample draws on
  with a weight above 0, so the 0s outside MoGe's mask never blend into the depth.
- view_prior (Req 10.8): a Prior_View's Prior as read, at the Prior_Size, and its
  Brightness_Gradient at the Training_Size, for the view's View_Record.

plain_pixels, depth_weight, upscale and prior_pixel_mask use only indexing, comparisons,
arithmetic and, through `xp`, exp and where, so they take numpy arrays on the CPU and torch
tensors on the GPU alike. The Trainer calls these same functions, so the rule the Test_Suite
tests is the rule the Trainer uses (Req 11.7). The taps are numpy arrays here, and the Trainer
can make a PriorTaps of tensors from them.

This module imports only the standard library, numpy and cv2, and from splatpipe only
splatpipe.train.priors and splatpipe.train.settings for their types. It imports no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from splatpipe.train.priors import Prior
from splatpipe.train.settings import TrainSettings

# A numpy array on the CPU, or a torch tensor on the GPU, of the same shape and meaning.
Array = Any

_GREY_RANGE = 255  # Fixed_Constant: an 8-bit grey value runs from 0 to 255 (Req 11.1, A12)


def brightness(pixels: np.ndarray) -> np.ndarray:
    """The Brightness of each training pixel, (H, W) float64 from 0 to 1, of the (H, W, 3) uint8
    RGB Training_Image at the Training_Size: OpenCV's grey value, / 255 (Req 11.1, A12)."""
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) / _GREY_RANGE


def brightness_gradient(grey: np.ndarray) -> np.ndarray:
    """The Brightness_Gradient of the (H, W) Brightness `grey`, as (H, W) float32 (Req 11.2, A12).

    With J the Brightness padded by one pixel that repeats the border outward, the gradient is
    (J[y, x + 1] − J[y, x − 1]) / 2 across and (J[y + 1, x] − J[y − 1, x]) / 2 down, and its
    length √(across² + down²) is worked out in float64, then rounded to float32 once.
    """
    padded = np.pad(grey, 1, mode="edge")
    across = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
    down = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
    return np.sqrt(across * across + down * down).astype(np.float32)


def plain_pixels(gradient: Array, settings: TrainSettings) -> Array:
    """The Plain_Pixels, True where the Brightness_Gradient is below plain_gradient (Req 11.3)."""
    return gradient < settings.plain_gradient


def depth_weight(rate: float, gradient: Array, settings: TrainSettings, xp: Any = np) -> Array:
    """Each training pixel's Depth_Weight at an Iteration whose Depth_Schedule value is `rate`
    (Req 11.5, D2).

    It's rate × exp(−depth_gradient_falloff × gradient), and a Plain_Pixel whose product is below
    depth_weight_floor gets the floor. So a Plain_Pixel's weight is the larger of the two, and a
    textured pixel's can go lower. `xp` is numpy, or torch for a tensor `gradient`: the function
    uses only xp.exp, xp.where and operators, and passes the floor as a Python scalar.
    """
    product = rate * xp.exp(-settings.depth_gradient_falloff * gradient)
    lifted = plain_pixels(gradient, settings) & (product < settings.depth_weight_floor)
    return xp.where(lifted, settings.depth_weight_floor, product)


@dataclass(frozen=True)
class PriorTaps:
    """The scale-up from the Prior_Size to the Training_Size H × W, one axis each (Req 10.6).

    Each axis holds i0 and i1, the two stored pixels a training pixel's bilinear sample draws on,
    and w1, the weight of i1, so i0's weight is 1 − w1. The nearest stored pixels are the ones
    under each training pixel's centre, which the mask is sampled at.
    """

    rows: tuple[np.ndarray, np.ndarray, np.ndarray]  # i0, i1 int64 (H,), w1 float32 (H,)
    columns: tuple[np.ndarray, np.ndarray, np.ndarray]  # i0, i1 int64 (W,), w1 float32 (W,)
    nearest_rows: np.ndarray  # (H,) int64
    nearest_columns: np.ndarray  # (W,) int64


def upscale_taps(stored: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bilinear taps of one axis, from `stored` pixels to `target` pixels, as (i0, i1, w1):
    int64, int64 and float32 arrays of `target` values (Req 10.6).

    Training pixel X sits at stored coordinate (X + 0.5) · S / T − 0.5, which is q / (2T) with
    q = (2X + 1) · S − T, worked out in integers. Where q ≤ 0, i0 is 0 and w1 is 0, the clamp at
    the first pixel. Elsewhere i0 is q // (2T) and w1 is (q mod 2T) / (2T), rounded to float32.
    i1 is min(i0 + 1, S − 1), and w1 is 0 where i1 is i0, the clamp at the last pixel.
    """
    span = 2 * target
    q = (2 * np.arange(target, dtype=np.int64) + 1) * stored - target
    inside = q > 0
    first = np.where(inside, q // span, 0)
    remainder = np.where(inside, q % span, 0)
    second = np.minimum(first + 1, stored - 1)
    remainder[second == first] = 0
    return first, second, (remainder / span).astype(np.float32)


def _nearest(stored: int, target: int) -> np.ndarray:
    """The stored pixel under each of `target` training pixels' centres, (2X + 1) · S // (2T),
    as int64 (Req 10.6)."""
    return (2 * np.arange(target, dtype=np.int64) + 1) * stored // (2 * target)


def prior_taps(prior_width: int, prior_height: int, width: int, height: int) -> PriorTaps:
    """The PriorTaps from the Prior_Size `prior_width` × `prior_height` to the Training_Size
    `width` × `height`."""
    return PriorTaps(rows=upscale_taps(prior_height, height),
                     columns=upscale_taps(prior_width, width),
                     nearest_rows=_nearest(prior_height, height),
                     nearest_columns=_nearest(prior_width, width))


def _grid(taps: PriorTaps) -> tuple[Array, Array, Array, Array]:
    """The four tap index arrays shaped to broadcast to (H, W): i0 and i1 of the rows as (H, 1),
    and of the columns as (1, W)."""
    (r0, r1, _), (c0, c1, _) = taps.rows, taps.columns
    return r0[:, None], r1[:, None], c0[None, :], c1[None, :]


def upscale(stored: Array, taps: PriorTaps) -> Array:
    """The (h, w) or (h, w, 3) Prior array `stored`, sampled bilinearly at the Training_Size H × W
    (Req 10.6).

    The four taps are gathered with broadcast index arrays and blended across, then down:
    top = a·(1 − wc) + b·wc, bottom = c·(1 − wc) + d·wc and value = top·(1 − wr) + bottom·wr, with
    the weights shaped to broadcast over a channel axis.
    """
    rows0, rows1, columns0, columns1 = _grid(taps)
    channels = (1,) * (stored.ndim - 2)
    down = taps.rows[2].reshape((-1, 1, *channels))
    across = taps.columns[2].reshape((1, -1, *channels))
    top = stored[rows0, columns0] * (1 - across) + stored[rows0, columns1] * across
    bottom = stored[rows1, columns0] * (1 - across) + stored[rows1, columns1] * across
    return top * (1 - down) + bottom * down


def prior_pixel_mask(mask: Array, keep: Array | None, taps: PriorTaps) -> Array:
    """The (H, W) bool Prior_Pixels of a view whose Prior has the (h, w) bool `mask` (Req 10.7).

    A training pixel is a Prior_Pixel when it isn't a Masked_Pixel, True in `keep`, or any pixel
    when `keep` is None because the view has no Mask_File; when the stored pixel under its centre
    has a Prior; and when every stored pixel its bilinear sample draws on with a weight above 0
    has one. i0's weight, 1 − w1, is always above 0, and i1's is w1, so a stored pixel with no
    Prior and a weight of 0 doesn't count (A13).
    """
    rows0, rows1, columns0, columns1 = _grid(taps)
    no_down = (taps.rows[2] == 0)[:, None]
    no_across = (taps.columns[2] == 0)[None, :]
    found = (mask[rows0, columns0]
             & (no_across | mask[rows0, columns1])
             & (no_down | mask[rows1, columns0])
             & (no_down | no_across | mask[rows1, columns1])
             & mask[taps.nearest_rows[:, None], taps.nearest_columns[None, :]])
    return found if keep is None else found & keep


@dataclass(frozen=True)
class ViewPrior:
    """A Prior_View's Prior and Brightness_Gradient, kept in its View_Record (Req 10.8)."""

    prior: Prior  # at the Prior_Size, as read
    gradient: np.ndarray  # (H, W) float32, the Brightness_Gradient at the Training_Size


def view_prior(pixels: np.ndarray, prior: Prior) -> ViewPrior:
    """The ViewPrior of a Prior_View whose (H, W, 3) uint8 RGB pixels at the Training_Size are
    `pixels`: its Prior as read and its Brightness_Gradient."""
    return ViewPrior(prior, brightness_gradient(brightness(pixels)))
