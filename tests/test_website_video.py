"""The Box_Reader and the Video_Check, over small MP4 and MOV files and a sparse reader."""

from __future__ import annotations

import pytest

from app.errors import PublicRefusal
from app.video import BROKEN, GONE, INCOMPLETE, NO_VIDEO, check_video, find_moov, read_movie
from tests.website_helpers import box, mp4, real_config, website_config

JOBS = real_config().jobs()


class Reader:
    """Serves `data`, records each read, and can fail a read inside another Box's payload."""

    def __init__(self, data: bytes, forbidden: list[tuple[int, int]] | None = None) -> None:
        self.data = data
        self.reads: list[tuple[int, int]] = []
        self.forbidden = forbidden or []

    def __call__(self, offset: int, length: int) -> bytes:
        self.reads.append((offset, length))
        for start, end in self.forbidden:
            assert offset + length <= start or offset >= end, (offset, length)
        return self.data[offset:offset + length]


def check(data: bytes, name: str = "IMG_0042.MOV", **kwargs):
    return check_video(Reader(data), len(data), len(data), name, website_config(**kwargs), JOBS)


@pytest.mark.parametrize("moov_first", [True, False])
def test_the_moov_is_found_before_or_after_the_media(moov_first: bool) -> None:
    facts = check(mp4(12.5, moov_first=moov_first), "clip.mp4")
    assert (facts.video_file, facts.video_seconds) == ("video.mp4", 12.5)
    assert (facts.start_seconds, facts.end_seconds) == (0.0, 12.5)


def test_only_headers_are_read_outside_the_moov() -> None:
    data = mp4(8, mdat_bytes=4096)
    ftyp_len = int.from_bytes(data[:4], "big")
    mdat_len = int.from_bytes(data[ftyp_len:ftyp_len + 4], "big")
    payloads = [(8, ftyp_len), (ftyp_len + 8, ftyp_len + mdat_len)]
    reader = Reader(data, payloads)
    assert check_video(reader, len(data), len(data), "a.mov", website_config(), JOBS).video_seconds == 8


def test_a_64_bit_box_size_and_a_size_zero_last_box() -> None:
    data = mp4(6)
    moov_at = data.index(b"moov") - 4
    head, moov = data[:moov_at], data[moov_at:]
    big_mdat = box(b"free", bytes(100), large=True)
    rebuilt = head + big_mdat + moov
    assert check(rebuilt).video_seconds == 6
    to_end = rebuilt[:-len(moov)] + (0).to_bytes(4, "big") + moov[4:]
    assert check(to_end).video_seconds == 6


def test_the_track_length_falls_back_on_the_movie_header() -> None:
    facts = read_movie(mp4(9, track_duration=0)[mp4(9, track_duration=0).index(b"moov") - 4:])
    assert facts.has_video and facts.video_seconds == 9
    assert read_movie(mp4(9, version=1)[mp4(9, version=1).index(b"moov") - 4:]).video_seconds == 9


@pytest.mark.parametrize("mutate", [
    lambda d: d[:-3],                                         # the last Box runs past the end
    lambda d: (4).to_bytes(4, "big") + d[4:],                 # a Box smaller than its header
    lambda d: d.replace(b"moov", b"moof"),                    # no moov
    lambda d: b"\x00\x00",                                    # too short to hold a header
])
def test_broken_files_are_refused(mutate) -> None:
    with pytest.raises(PublicRefusal) as caught:
        check(mutate(mp4(5)))
    assert caught.value.status == 422 and caught.value.message == BROKEN


def test_too_many_boxes_and_a_huge_moov_are_refused() -> None:
    data = box(b"ftyp", b"isom") + box(b"free") * 5 + mp4(5)[mp4(5).index(b"moov") - 4:]
    with pytest.raises(PublicRefusal, match="can read"):
        find_moov(Reader(data), len(data), 4, 1 << 20)
    with pytest.raises(PublicRefusal, match="can read"):
        check(mp4(5), max_moov_bytes=16)


def test_the_facts_and_the_upload_are_checked() -> None:
    with pytest.raises(PublicRefusal, match="no video track"):
        check(mp4(5, video=False))
    with pytest.raises(PublicRefusal, match="longer than 10 minutes"):
        check(mp4(601))
    with pytest.raises(PublicRefusal, match="shorter than 3 s"):
        check(mp4(2))
    with pytest.raises(PublicRefusal, match="Only MP4 and MOV"):
        check(mp4(5), "clip.avi")
    data = mp4(5)
    with pytest.raises(PublicRefusal) as gone:
        check_video(Reader(data), None, len(data), "a.mp4", website_config(), JOBS)
    assert gone.value.message == GONE
    with pytest.raises(PublicRefusal) as short:
        check_video(Reader(data), len(data), len(data) + 1, "a.mp4", website_config(), JOBS)
    assert short.value.message == INCOMPLETE
    with pytest.raises(PublicRefusal, match="larger than 4 GB"):
        check_video(Reader(data), JOBS.max_video_bytes + 1, JOBS.max_video_bytes + 1, "a.mp4",
                    website_config(), JOBS)
    assert NO_VIDEO
