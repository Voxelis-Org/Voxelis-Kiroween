"""The Depth_Step's files: the Camera_Solve_Output it reads, and the Prior_Files and the
Prior_Index it writes (depth-priors spec, Req 2 and 7).

The Depth_Step reads the same files as the Train_Step, under the Train_Step's names (Req 2.1):

    work/cameras/images/<name>.jpg   the Frames, <name>.jpg being an image's name in images.bin
    work/cameras/sparse/             the Sparse_Model: cameras.bin, images.bin and points3D.bin

and writes the Prior_Format of splatpipe.train.priors in work/depth/:

    work/depth/<stem>.npz   one Prior_File per Fitted_Frame
    work/depth/depth.json   the Prior_Index, written last

check_inputs (Req 2.2) checks that images/ is a folder, then that each model file in sparse/ is a
file, and refuses the first one missing, naming it and the camera solve step. It reads nothing,
so the step runs it before the GPU checks.

read_frame decodes a Frame with the Train_Step's own read_training_image, so a Frame with no file,
one OpenCV can't decode and one that isn't the Shared_Camera's size are refused with the
Train_Step's messages (Req 2.7). It scales the Frame to the Prior_Size with the Train_Step's
scaled_down, INTER_AREA, when the Frame is larger, and turns it from cv2's BGR to RGB (Req 4.1,
4.2). The division by 255 happens on the GPU. cv2 only decodes still images and scales them
here, and never reads or writes video.

write_prior_file and write_prior_index call the Prior_Format's write_prior and write_index, and
turn an OSError into an ExpectedFailure that names the file and the reason, like "No space left
on device" (Req 7.10). write_index writes depth.json through a temporary file and one os.replace,
and removes the temporary file when that fails, so a reader never finds a half-written index and
a failed write leaves none (Req 7.6, 7.8).

From splatpipe, this module imports only splatpipe.errors, splatpipe.train.files and
splatpipe.train.priors, and splatpipe.train.colmap for its types.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.files import IMAGES, MODEL_FILES, SPARSE, read_training_image, scaled_down
from splatpipe.train.priors import (PRIOR_INDEX, Prior, PriorIndex, prior_path, write_index,
                                    write_prior)

if TYPE_CHECKING:
    from splatpipe.train.colmap import Camera


def check_inputs(cameras: Path) -> None:
    """Refuse a Camera_Solve_Output in `cameras`, work/cameras/, without images/ or a model file.

    images/ must be a folder and each of MODEL_FILES in sparse/ a file, checked in that order.
    The first one missing is refused with the Req 2.2 message, which names it. Nothing is read.
    """
    images = cameras / IMAGES
    if not images.is_dir():
        raise _missing(images)
    for name in MODEL_FILES:
        path = cameras / SPARSE / name
        if not path.is_file():
            raise _missing(path)


def read_frame(cameras: Path, name: str, camera: Camera, width: int, height: int) -> np.ndarray:
    """The Frame `name` in `cameras`, work/cameras/, as (height, width, 3) uint8 RGB at the
    Prior_Size `width` × `height` (Req 2.7, 4.1, 4.2).

    It's decoded and checked against the Shared_Camera `camera` with the Train_Step's
    read_training_image, and scaled down with INTER_AREA only when it's larger than the
    Prior_Size, which the Training_Size rule never makes larger than the Frame.
    """
    image = read_training_image(cameras / IMAGES, name, camera)
    return cv2.cvtColor(scaled_down(image, width, height), cv2.COLOR_BGR2RGB)


def write_prior_file(folder: Path, name: str, prior: Prior) -> None:
    """Write the image `name`'s `prior` to its Prior_File in `folder`, work/depth/ (Req 7.1).

    An OSError becomes the Req 7.10 failure, naming the file and the reason.
    """
    path = prior_path(folder, name)
    try:
        write_prior(path, prior)
    except OSError as error:
        raise _not_written(path, error) from error


def write_prior_index(folder: Path, index: PriorIndex) -> None:
    """Write `index` to the Prior_Index, depth.json in `folder`, work/depth/, last and through a
    temporary file (Req 7.3, 7.6).

    An OSError becomes the Req 7.10 failure, naming depth.json and the reason, and write_index has
    removed the temporary file by then, so no index is left.
    """
    try:
        write_index(folder, index)
    except OSError as error:
        raise _not_written(folder / PRIOR_INDEX, error) from error


def _missing(path: Path) -> ExpectedFailure:
    """The Req 2.2 failure for a path of the Camera_Solve_Output that isn't there."""
    return ExpectedFailure(f"The depth step needs {path}, which the camera solve step writes, but "
                           "it isn't there. Run the camera solve step on this job first.")


def _not_written(path: Path, error: OSError) -> ExpectedFailure:
    """The Req 7.10 failure for the file at `path`, which couldn't be written."""
    return ExpectedFailure(f"The file {path} couldn't be written ({_reason(error)}). Free some "
                           f"space on the disk, or check that you can write to {path.parent}, "
                           "then run the step again.")


def _reason(error: OSError) -> str:
    """The reason the system gave for an OSError, like "No space left on device"."""
    return error.strerror or str(error)
