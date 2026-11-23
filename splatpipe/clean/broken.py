"""The Broken_Filter: it drops the Gaussians that nothing can draw.

A Gaussian is broken when one of its eleven Required_Properties is a NaN or an infinity, or when
its rotation quaternion has zero length (Req 2.2). The filter runs first in the Filter_Order, on
every Gaussian of the Splat_Data, so the Faint_Filter and the Isolated_Filter only ever see finite
centres, opacities and scales. It reads no field outside REQUIRED, so a NaN or an infinity in a
colour or another Carried_Property stays, and write_splat carries it into the Clean_Splat
unchanged (Req 2.3, A11).

From splatpipe, this module imports only splatpipe.clean.ply_read.
"""

from __future__ import annotations

import numpy as np

from splatpipe.clean.ply_read import REQUIRED


def broken_keep(data: np.ndarray) -> np.ndarray:
    """Which Gaussians of `data` the Broken_Filter keeps, as a bool array with one per Gaussian.

    A Gaussian is kept when np.isfinite holds for every field in REQUIRED, which it always does
    for an integer field, and its quaternion isn't four zeros (Req 2.2). A quaternion's length is
    0 exactly when all four components are 0.0 or -0.0, so each one is tested with == 0 and
    nothing is squared: a square overflows float32 for a component above about 1.8e19, which
    would raise under the tests' error settings. The rule looks at each Gaussian's own
    Required_Properties alone, so running it again on what it kept keeps every one (Req 2.3,
    2.19).

    The mask is built in place, one field at a time, so besides it only a few bool temporaries
    of one byte a Gaussian are alive at once.
    """
    keep = np.ones(len(data), dtype=bool)
    for name in REQUIRED:
        keep &= np.isfinite(data[name])
    keep &= ~((data["rot_0"] == 0) & (data["rot_1"] == 0) & (data["rot_2"] == 0)
              & (data["rot_3"] == 0))
    return keep
