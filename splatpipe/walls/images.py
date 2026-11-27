"""The Walls_Step's score images and window sums, as pure numpy and OpenCV functions (walls spec,
Req 9.1, 9.4, 10.5).

The score images, at the Score_Size, in float64:
- luma: the Frame's grey, OpenCV's BGR-to-grey conversion, from 0 to 255. The Frame is turned
  into float32 before the conversion, so the grey isn't rounded to whole levels: rounding it
  would add a noise of variance 1/12 ≈ 0.083 to every pixel, more than the flat test's
  `flat_variance`, and faint wall texture would drown in it.
- high_pass: the Luma minus its Gaussian blur at `high_pass_sigma`, which the First_Pass and the
  Second_Pass score and every flat test reads (A2).
- bands: the octave Bands, blur(σₖ) − blur(σₖ₊₁) for each pair of neighbouring `band_sigmas`, a
  σ of 0 being the Luma itself, so [0, 1, 2, 4, 8] gives 4 Bands, which add up to the Luma minus
  its blur at the last σ.
- band_weights (Req 10.5, A4): each Band's w = √(ρ / ((1 − ρ²) · var)), so w² · var = ρ / (1 − ρ²),
  with its Coherence ρ capped at `band_max_coherence`, and 0 where ρ isn't above 0 or var is 0.
- band_mix: Σ w_b · Band_b, which the Third_Pass scores, for the Prior_Frame and for each
  Source_Frame with the Prior_Frame's own weights (Req 10.6).

The windows (Req 9.2, 9.4):
- window_centres: the grid every `window_step` pixels across and down, starting
  ⌊`window_size` / 2⌋ in from the top left, keeping the windows that lie inside the image.
- window_sums: a float image's sum over each window, from cv2.boxFilter(..., CV_64F,
  normalize=False) read at the centres, or a bool image's exact count of true pixels in each
  window, from corner reads of an integral image, for the masked and invalid pixels.
- window_variance: a window's variance per pixel, (S_II − S_I² / n) / n, the flat test's number.
- warp_image: the Source_Frame's image sampled at the warp's maps (geometry.warp_maps) with
  cv2.remap, INTER_LINEAR, BORDER_CONSTANT 0, as Req 9.3 samples it.
- window_ncc (Req 9.4): each window's zero-mean normalised cross-correlation of the Prior_Frame's
  image and the warped Source_Frame's, cov / √(var_I · var_J) from the five window sums, with
  var_I = S_II − S_I² / n, var_J = S_JJ − S_J² / n, cov = S_IJ − S_I · S_J / n, n = window_size².

The maps. warp_maps gives float64 maps and cv2.remap takes float32 ones, which hold a sample
position at 640 pixels to about 3e-5 of a pixel. That doesn't limit the sampling: with float maps
and INTER_LINEAR, cv2.remap rounds each sample position to the nearest 1/32 of a pixel (OpenCV's
INTER_BITS) before it interpolates, whatever the maps' precision, the two-channel map form too.
So the maps go to cv2.remap as two one-channel float32 arrays. A pixel that isn't `inside` the
Source_Frame is sampled off the image, so it reads the border's 0, and a window that holds one
gets an NCC of 0.

A variance counts as 0 when it is within the rounding of its sums: boxFilter keeps running sums
along the rows and down the columns, so each sum carries an error of a few ε times the largest
window sum of the image for each step along the image. A flat window's variance is then exactly
0, not a rounding error of either sign that would make its NCC anything from −1 to 1.

Every function is elementwise or a single pass of numpy or OpenCV. cv2.cvtColor, GaussianBlur,
boxFilter and remap give the same bits on one thread as on many, and the integral images are
numpy's sequential cumsum, so the results don't depend on the worker count (C14). This module
never sets OpenCV's thread count: the Worker_Pool does. It imports only the standard library,
numpy and cv2, and cv2 only converts colours, blurs, box-filters and remaps images.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Centres = tuple[np.ndarray, np.ndarray]  # (centres_x, centres_y), (P,) int, in pixels

# Where warp_image sends a pixel that isn't inside the Source_Frame: far enough off the image
# that both of each axis's bilinear taps read the border's 0.
_OFF_IMAGE = -4.0
# A few ε per step of boxFilter's running sums: a variance within this many ε × (the image's
# width + height + the window's edge) × the largest window sum of squares is rounding, so 0.
_ROUNDING_STEPS = 8.0


def luma(bgr: np.ndarray) -> np.ndarray:
    """The Luma of the (h, w, 3) BGR Frame `bgr`, as (h, w) float64 from 0 to 255.

    cv2.cvtColor(..., COLOR_BGR2GRAY), OpenCV's grey conversion, of the Frame turned into
    float32, so the grey keeps its fractions of a level.
    """
    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"luma: wants an (h, w, 3) BGR image, got shape {image.shape}")
    grey = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.float32), cv2.COLOR_BGR2GRAY)
    return grey.astype(np.float64)


def _blur(grey: np.ndarray, sigma: float) -> np.ndarray:
    """`grey` blurred with a Gaussian of `sigma` pixels, OpenCV's kernel size for it, or `grey`
    itself for a sigma of 0."""
    if sigma == 0:
        return grey
    return cv2.GaussianBlur(grey, (0, 0), float(sigma))


def _grey(image: np.ndarray) -> np.ndarray:
    """`image` as a contiguous 2-D float64 array, refusing any other shape."""
    grey = np.ascontiguousarray(image, dtype=np.float64)
    if grey.ndim != 2:
        raise ValueError(f"wants a 2-D grey image, got shape {grey.shape}")
    return grey


def high_pass(grey: np.ndarray, sigma: float) -> np.ndarray:
    """The High_Pass of the Luma `grey`: `grey` minus cv2.GaussianBlur(grey, (0, 0), sigma), with
    `sigma`, high_pass_sigma, in Score_Size pixels, as float64."""
    grey = _grey(grey)
    return grey - _blur(grey, sigma)


def bands(grey: np.ndarray, sigmas: Sequence[float]) -> list[np.ndarray]:
    """The octave Bands of the Luma `grey`: blur(σₖ) − blur(σₖ₊₁) for each pair of neighbouring
    `sigmas`, band_sigmas, a σ of 0 being `grey` itself, as float64. [0, 1, 2, 4, 8] gives 4."""
    if len(sigmas) < 2:
        raise ValueError(f"bands: wants at least 2 sigmas, got {list(sigmas)}")
    grey = _grey(grey)
    blurs = [_blur(grey, float(sigma)) for sigma in sigmas]
    return [blurs[k] - blurs[k + 1] for k in range(len(blurs) - 1)]


def band_weights(coherences: Sequence[float] | np.ndarray, variances: Sequence[float] | np.ndarray,
                 cap: float) -> np.ndarray:
    """Each Band's Band_Weight, as (B,) float64 (Req 10.5, A4).

    Its Coherence ρ capped at `cap`, band_max_coherence, gives w = √(ρ / ((1 − ρ²) · var)) with
    var its variance over the Prior_Frame's Depth_Pixels no mask hides, so w² · var = ρ / (1 − ρ²).
    A Band whose ρ isn't above 0 (NaN included) or whose var isn't a finite number above 0 gets 0.
    """
    rho = np.asarray(coherences, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    if rho.shape != var.shape or rho.ndim != 1:
        raise ValueError(f"band_weights: wants one coherence and one variance per Band, got "
                         f"shapes {rho.shape} and {var.shape}")
    with np.errstate(invalid="ignore"):
        good = (rho > 0) & np.isfinite(var) & (var > 0)
    capped = np.minimum(np.where(good, rho, 0.0), float(cap))
    safe_var = np.where(good, var, 1.0)
    weights = np.sqrt(capped / ((1.0 - capped * capped) * safe_var))
    return np.where(good, weights, 0.0)


def band_mix(band_images: Sequence[np.ndarray] | np.ndarray,
             weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """The Band_Mix Σ w_b · Band_b of the Bands `band_images`, a list or a (B, h, w) array, with
    the Band_Weights `weights`, as (h, w) float64, summed elementwise in Band order."""
    weights = np.asarray(weights, dtype=np.float64)
    if len(band_images) != weights.shape[0] or len(band_images) == 0:
        raise ValueError(f"band_mix: wants one weight per Band, got {len(band_images)} Bands and "
                         f"{weights.shape[0]} weights")
    mix = float(weights[0]) * np.asarray(band_images[0], dtype=np.float64)
    for weight, band in zip(weights[1:], band_images[1:]):
        mix = mix + float(weight) * np.asarray(band, dtype=np.float64)
    return mix


def window_centres(width: int, height: int, window_size: int, step: int) -> Centres:
    """The window centres of a `width` × `height` image, as (centres_x, centres_y), (P,) int64:
    every `step` pixels across and down from ⌊`window_size` / 2⌋ in from the top left, keeping
    the windows that lie inside the image, in row-major order (Req 9.2, A2)."""
    half = window_size // 2
    xs = np.arange(half, width - half, step, dtype=np.int64)
    ys = np.arange(half, height - half, step, dtype=np.int64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_x.ravel(), grid_y.ravel()


def _check_windows(shape: tuple[int, ...], centres: Centres, window_size: int) -> tuple[
        np.ndarray, np.ndarray]:
    """The centres as int64 arrays, refusing an even window, a mismatch of the two arrays and
    any window that leaves an image of `shape`."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"wants an odd window size, got {window_size}")
    xs = np.asarray(centres[0], dtype=np.int64)
    ys = np.asarray(centres[1], dtype=np.int64)
    if xs.shape != ys.shape:
        raise ValueError(f"wants as many x as y centres, got {xs.shape} and {ys.shape}")
    half = window_size // 2
    height, width = shape[:2]
    if xs.size and (xs.min() < half or xs.max() > width - 1 - half
                    or ys.min() < half or ys.max() > height - 1 - half):
        raise ValueError(f"a {window_size}-pixel window leaves the {width} × {height} image")
    return xs, ys


def window_sums(image: np.ndarray, centres: Centres, window_size: int) -> np.ndarray:
    """Each window's sum of `image`, (h, w), as (P,): the `window_size` × `window_size` window
    around each centre of `centres`, which window_centres gives.

    A bool image gives each window's exact count of true pixels, as int64, from corner reads of
    an integral image: the masked or invalid pixels a window holds. Any other image gives float64
    sums, cv2.boxFilter(image, CV_64F, (W, W), normalize=False) read at the centres.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"window_sums: wants a 2-D image, got shape {image.shape}")
    xs, ys = _check_windows(image.shape, centres, window_size)
    half = window_size // 2
    if image.dtype == np.bool_:
        integral = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.int64)
        integral[1:, 1:] = np.cumsum(np.cumsum(image, axis=0, dtype=np.int64), axis=1)
        top, bottom = ys - half, ys + half + 1
        left, right = xs - half, xs + half + 1
        return (integral[bottom, right] - integral[top, right]
                - integral[bottom, left] + integral[top, left])
    sums = cv2.boxFilter(np.ascontiguousarray(image, dtype=np.float64), cv2.CV_64F,
                         (window_size, window_size), normalize=False)
    return sums[ys, xs]


def _rounding(image: np.ndarray, window_size: int) -> float:
    """How far a window's S_II − S_I² / n may be off from rounding alone, for `image`: a few ε for
    each step of boxFilter's running sums, times the largest window sum of squares it could
    hold."""
    height, width = image.shape
    largest = float(np.max(image * image)) * window_size * window_size if image.size else 0.0
    steps = width + height + window_size
    return _ROUNDING_STEPS * float(np.finfo(np.float64).eps) * steps * largest


def _variances(image: np.ndarray, centres: Centres, window_size: int) -> tuple[
        np.ndarray, np.ndarray]:
    """Each window's S_I and S_II − S_I² / n of the float64 `image`, the second set to 0 where it
    is within rounding of 0."""
    n = float(window_size * window_size)
    s_i = window_sums(image, centres, window_size)
    s_ii = window_sums(image * image, centres, window_size)
    spread = s_ii - s_i * s_i / n
    return s_i, np.where(spread > _rounding(image, window_size), spread, 0.0)


def window_variance(image: np.ndarray, centres: Centres, window_size: int) -> np.ndarray:
    """Each window's variance per pixel of `image`, (S_II − S_I² / n) / n with n = window_size²,
    as (P,) float64, and exactly 0 where it is within rounding of 0. The flat test compares the
    High_Pass's with `flat_variance` (Req 9.2, 9.5, A2)."""
    _, spread = _variances(_grey(image), centres, window_size)
    return spread / float(window_size * window_size)


def warp_image(source_image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
               inside: np.ndarray) -> np.ndarray:
    """The Source_Frame's `source_image` sampled at the warp's maps, as float64 in the maps'
    shape: cv2.remap(..., INTER_LINEAR, borderMode=BORDER_CONSTANT) with the border 0 (Req 9.3).

    `map_x`, `map_y` and `inside` are geometry.warp_maps's: OpenCV's sample coordinates and where
    the point lands in front of the Source_Frame and inside it. The maps go to cv2.remap as
    float32, and a pixel that isn't `inside`, or whose map isn't finite, is sampled off the image,
    so it reads 0. A bool image is sampled as 0 and 1: warping its masked pixels as 1 gives
    exactly 0 where every tap a pixel reads is kept, and more than 0 where one is masked.
    """
    source = np.ascontiguousarray(source_image, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"warp_image: wants a 2-D image, got shape {source.shape}")
    inside = np.asarray(inside, dtype=bool)
    map_x = np.asarray(map_x, dtype=np.float64)
    map_y = np.asarray(map_y, dtype=np.float64)
    usable = inside & np.isfinite(map_x) & np.isfinite(map_y)
    sample_x = np.where(usable, map_x, _OFF_IMAGE).astype(np.float32)
    sample_y = np.where(usable, map_y, _OFF_IMAGE).astype(np.float32)
    return cv2.remap(source, sample_x, sample_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def window_ncc(prior_image: np.ndarray, source_image: np.ndarray, map_x: np.ndarray,
               map_y: np.ndarray, inside: np.ndarray, centres: Centres, window_size: int,
               flat_variance: float) -> np.ndarray:
    """Each window's Window_Score, the zero-mean normalised cross-correlation of the Prior_Frame's
    `prior_image` and the Source_Frame's `source_image` warped at the maps, as (P,) float64 from
    −1 to 1 (Req 9.4).

    The source is sampled with warp_image. With n = window_size² and the five window sums
    S_I, S_J, S_II, S_JJ and S_IJ, var_I = S_II − S_I² / n, var_J = S_JJ − S_J² / n and
    cov = S_IJ − S_I · S_J / n, the score is cov / √(var_I · var_J). It is 0 where either variance
    is 0, or below `flat_variance` per pixel, or where a pixel of the window isn't `inside`. Give
    a `flat_variance` of 0 to score every window with any variance; the Valid_Pair tests are the
    scoring's, which reads the High_Pass's variance for them (A2).
    """
    prior = _grey(prior_image)
    inside = np.asarray(inside, dtype=bool)
    if prior.shape != inside.shape or np.shape(map_x) != prior.shape or np.shape(
            map_y) != prior.shape:
        raise ValueError(f"window_ncc: the maps and `inside` must be the Prior_Frame's shape "
                         f"{prior.shape}")
    warped = warp_image(source_image, map_x, map_y, inside)
    n = float(window_size * window_size)
    s_i, var_i = _variances(prior, centres, window_size)
    s_j, var_j = _variances(warped, centres, window_size)
    s_ij = window_sums(prior * warped, centres, window_size)
    cov = s_ij - s_i * s_j / n
    floor = float(flat_variance) * n
    outside = window_sums(~inside, centres, window_size) > 0
    good = (var_i > 0) & (var_j > 0) & (var_i >= floor) & (var_j >= floor) & ~outside
    with np.errstate(divide="ignore", invalid="ignore"):
        score = cov / np.sqrt(np.where(good, var_i * var_j, 1.0))
    return np.where(good, np.clip(score, -1.0, 1.0), 0.0)
