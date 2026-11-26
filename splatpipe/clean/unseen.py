"""The Unseen_Filter: it drops the Gaussians that too few training frames saw.

The training step trains each Gaussian only through the frames whose camera sees it, so a
Gaussian no frame sees, like the fog that grows above a room, is never checked against a
picture. A frame sees a Gaussian when the Gaussian's centre is in front of the frame's camera
and lands inside the frame. A Gaussian is kept when at least min_seen_frames frames see it.

The frames are the training frames of the camera solve: each image of the Sparse_Model in
work/cameras/sparse/, with its world-to-camera pose and the one shared PINHOLE camera at its
full size, W x H. The training step trains in that model's frame and units, so the trained
splat's centres are in the same frame. A centre x is at c = R x + t in the camera. It's in front
when c_z > 0, and it lands inside the frame when its pixel, (fx c_x / c_z + cx, fy c_y / c_z +
cy), is in [0, W) x [0, H), the pixel grid COLMAP's PINHOLE camera covers. The test is worked
out with c_z multiplied through, as 0 <= fx c_x + cx c_z < W c_z and the same for y, so no
division can overflow or divide by 0, and float64 holds every product of a finite float32
centre. The training masks aren't used: a frame sees a Gaussian behind its mask too.

Each frame only checks the Gaussians that fewer than min_seen_frames frames have seen so far, so
most Gaussians are done after the first few frames.

`read_views` reads the Sparse_Model with the training step's Model_Reader, so a broken model is
refused with that reader's messages. From splatpipe, this module imports only splatpipe.errors
and the training step's colmap.py, files.py and scene.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import SparseModel, read_model
from splatpipe.train.files import MODEL_FILES
from splatpipe.train.scene import world_to_camera


@dataclass(frozen=True)
class Views:
    """The training frames' cameras: each frame's world-to-camera rotation and translation, and
    the shared PINHOLE camera at its full size."""

    rotations: np.ndarray  # (F, 3, 3) float64
    translations: np.ndarray  # (F, 3) float64
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def read_views(sparse: Path) -> Views:
    """The Views of the Sparse_Model in `sparse`, work/cameras/sparse/.

    Each of its three files must be a file, and the first one missing fails the step with a
    message that names it and the camera solve step. A model the Model_Reader refuses fails with
    the reader's own message.
    """
    for name in MODEL_FILES:
        path = sparse / name
        if not path.is_file():
            raise ExpectedFailure(f"The clean step needs {path}, which the camera solve step "
                                  "writes, but it isn't there. Run the camera solve step on this "
                                  "job first.")
    return views_of(read_model(sparse))


def views_of(model: SparseModel) -> Views:
    """The Views of `model`, one for each of its images, in images.bin's order."""
    matrices = [world_to_camera(image.quaternion, image.translation) for image in model.images]
    camera = model.camera
    return Views(rotations=np.array([m[:3, :3] for m in matrices]).reshape(-1, 3, 3),
                 translations=np.array([m[:3, 3] for m in matrices]).reshape(-1, 3),
                 width=camera.width, height=camera.height,
                 fx=camera.fx, fy=camera.fy, cx=camera.cx, cy=camera.cy)


def unseen_keep(points: np.ndarray, views: Views, min_seen: int) -> np.ndarray:
    """A bool mask over `points`, an (n, 3) float64 array of finite centres: True for each centre
    that at least `min_seen` of the frames in `views` see, as the module says."""
    seen = np.zeros(len(points), dtype=np.int64)
    pending = np.arange(len(points))
    for rotation, translation in zip(views.rotations, views.translations):
        if pending.size == 0:
            break
        seen[pending[_inside(points[pending], rotation, translation, views)]] += 1
        pending = pending[seen[pending] < min_seen]
    return seen >= min_seen


def _inside(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
            views: Views) -> np.ndarray:
    """Whether each of `points` is in front of the camera (rotation, translation) and lands in
    its frame, tested with c_z multiplied through, as the module says."""
    camera = points @ rotation.T + translation
    x, y, z = camera[:, 0], camera[:, 1], camera[:, 2]
    u = views.fx * x + views.cx * z
    v = views.fy * y + views.cy * z
    return ((z > 0) & (u >= 0) & (u < views.width * z) & (v >= 0) & (v < views.height * z))
