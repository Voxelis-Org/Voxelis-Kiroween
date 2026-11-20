"""JSON files that read back exactly as they were written.

`write_json` checks the whole value with `check_json_value` and only then writes plain
ASCII JSON with LF line endings. So `read_json` gives back an equal value, with the same
types all the way down, the same key order and the same float signs. Each refusal is a
`ValueError`. A refused value raises `NotJsonError`, whose message names the field.

This module imports no other splatpipe module.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

# The exact types check_json_value accepts, for its messages.
_ALLOWED = "dict with str keys, list, str, int, finite float, bool or None"

# json writes surrogates as escapes, and json.loads joins a high surrogate's escape followed
# by a low surrogate's into one character. So a str that holds the two as separate characters
# would read back changed, and no JSON text can keep them apart.
_SPLIT_PAIR = re.compile(r"[\ud800-\udbff][\udc00-\udfff]")
_SPLIT_PAIR_PROBLEM = ("holds a high surrogate followed by a low surrogate, which a JSON file "
                       "would read back as one character. Remove or replace the pair first.")

_FINITE_ONLY = "Our JSON files hold only finite numbers, so write the file again without it."


class NotJsonError(ValueError):
    """A value that isn't a JSON_Value. The message names the field, like `counts.frames` or `settings.sizes[2]`."""


def check_json_value(value: object, field: str) -> None:
    """Raise NotJsonError, naming the field path, unless `value` is a JSON_Value.

    Only the exact types dict with str keys, list, str, int, finite float, bool and None
    pass, nested in any way. Tuples, sets, subclasses such as IntEnum and look-alikes such
    as numpy's int64 are refused, and so is a str or key that holds a high surrogate followed
    by a low one, because json would read those back changed. An int with more digits than
    Python writes as text (sys.get_int_max_str_digits(), 4300 by default) is refused too,
    because json.dumps would fail on it with an error that names no field. `field` names the
    value. A dict key adds `.key` and a list index adds `[i]`, like `settings.a[1].b`. A value
    nested too deeply, such as a list that holds itself, raises RecursionError.
    """
    if type(value) is dict:
        _check_dict(value, field)
    elif type(value) is list:
        _check_list(value, field)
    elif type(value) is str:
        _check_text(value, field)
    elif type(value) is float:
        _check_float(value, field)
    elif type(value) is int:
        _check_int(value, field)
    elif not (value is None or type(value) is bool):
        raise NotJsonError(f"{field} has the type {type(value).__name__}, which our JSON files "
                           f"don't allow. Convert it first to one of these exact types: {_ALLOWED}.")


def _check_dict(value: dict[object, object], field: str) -> None:
    for key, item in value.items():
        _check_key(key, field)
        check_json_value(item, f"{field}.{key}")


def _check_key(key: object, field: str) -> None:
    if type(key) is not str:
        raise NotJsonError(f"A key in {field} has the type {type(key).__name__}, which our JSON "
                           "files don't allow. Make every key a str first.")
    _check_text(key, f"A key in {field}")


def _check_list(value: list[object], field: str) -> None:
    for index, item in enumerate(value):
        check_json_value(item, f"{field}[{index}]")


def _check_text(value: str, field: str) -> None:
    if _SPLIT_PAIR.search(value):
        raise NotJsonError(f"{field} {_SPLIT_PAIR_PROBLEM}")


def _check_float(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise NotJsonError(f"{field} is {value!r}, but our JSON files hold only finite numbers. "
                           "Use a finite number instead, or None for no value.")


def _check_int(value: int, field: str) -> None:
    limit = sys.get_int_max_str_digits()  # 0 means no limit
    # str() would raise on a long int, so compare with 10 ** limit, which has limit + 1 digits.
    # An int under 2 ** (3 * limit), which is 8 ** limit, has at most limit digits, so a normal
    # int stops at bit_length(). Python doesn't count the sign, hence abs().
    if limit and value.bit_length() > 3 * limit and abs(value) >= 10 ** limit:
        raise NotJsonError(f"{field} is an int with more than {limit} digits, which is more than "
                           "Python will write as text. Use a smaller number instead.")


def write_json(path: Path, value: object) -> None:
    """Check the whole value, then write it to `path` as indented ASCII JSON with LF line endings.

    The check names fields from the file's stem, like `fake.counts.frames` for `fake.json`.
    If it refuses the value, nothing is written and no file is made.
    """
    check_json_value(value, path.stem)
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    """Read a JSON file, refusing NaN, Infinity, -Infinity and numbers too big for a float.

    Every refusal of the file's content is a ValueError: UnicodeDecodeError for bytes that
    aren't UTF-8, json.JSONDecodeError for text that isn't JSON, and NotJsonError for a
    number that isn't finite. A file that can't be opened raises OSError, and JSON nested
    too deeply raises RecursionError.
    """
    text = path.read_text(encoding="utf-8")
    return json.loads(text, parse_constant=_refuse_constant, parse_float=_finite_float)


def _refuse_constant(name: str) -> NoReturn:
    """json.loads calls this for NaN, Infinity and -Infinity, which aren't JSON_Values."""
    raise NotJsonError(f"The file holds {name}. {_FINITE_ONLY}")


def _finite_float(text: str) -> float:
    """json.loads calls this for each number with a fraction or an exponent, like 1e400."""
    number = float(text)
    if not math.isfinite(number):
        raise NotJsonError(f"The file holds a number too big for a float. {_FINITE_ONLY}")
    return number
