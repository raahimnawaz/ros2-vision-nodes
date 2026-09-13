"""Tests for the cv_bridge replacement.

Runs with nothing but numpy and opencv-python-headless -- no rclpy, no ROS
distro, no camera. That is the point: this conversion sits on the hot path of
every image node, and it replaced a compiled dependency precisely so it could be
covered in CI on a plain ubuntu runner.

A fake message stands in for ``sensor_msgs/Image``. ``image_bridge`` is
duck-typed and reads only these five fields, so the fake is not a mock of the
real thing so much as the actual contract.
"""

import os
import sys

import numpy as np
import pytest

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PKG_DIR)

_REPO = os.path.abspath(os.path.join(_PKG_DIR, "..", "..", ".."))
os.environ.setdefault("GESTURE_BOT_SRC", _REPO)

from gesture_bot_ros.image_bridge import (  # noqa: E402
    UnsupportedEncoding,
    fill_bgr8,
    to_bgr8,
)


class FakeImage:
    """Everything image_bridge reads off a sensor_msgs/Image, and nothing else."""

    def __init__(self, height=0, width=0, encoding="", step=0, data=b""):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step
        self.data = data
        self.is_bigendian = 0
        self.header = None


def _msg(img, encoding, step=None):
    channels = 1 if img.ndim == 2 else img.shape[2]
    return FakeImage(
        height=img.shape[0],
        width=img.shape[1],
        encoding=encoding,
        step=step if step is not None else img.shape[1] * channels,
        data=np.ascontiguousarray(img).tobytes(),
    )


def _gradient(h=12, w=16, c=3):
    return np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)


# --------------------------------------------------------------- round trips
def test_bgr8_round_trips_exactly():
    img = _gradient()
    assert np.array_equal(to_bgr8(_msg(img, "bgr8")), img)


def test_rgb8_channels_are_swapped_not_passed_through():
    """The bug this guards: treating rgb8 as bgr8 gives a plausible image with
    red and blue exchanged, which is easy to miss on a hand-tracking demo."""
    img = _gradient()
    out = to_bgr8(_msg(img, "rgb8"))
    assert np.array_equal(out[:, :, 0], img[:, :, 2])
    assert np.array_equal(out[:, :, 1], img[:, :, 1])
    assert np.array_equal(out[:, :, 2], img[:, :, 0])


def test_mono8_expands_to_three_equal_channels():
    gray = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)
    out = to_bgr8(_msg(gray, "mono8"))
    assert out.shape == (12, 16, 3)
    assert np.array_equal(out[:, :, 0], gray)
    assert np.array_equal(out[:, :, 0], out[:, :, 2])


def test_bgra8_drops_alpha_and_keeps_channel_order():
    bgra = _gradient(c=4)
    out = to_bgr8(_msg(bgra, "bgra8"))
    assert out.shape == (12, 16, 3)
    assert np.array_equal(out, bgra[:, :, :3])


def test_encoding_is_case_insensitive():
    img = _gradient()
    assert np.array_equal(to_bgr8(_msg(img, "BGR8")), img)


# -------------------------------------------------------------- row padding
def test_padded_rows_are_not_sheared():
    """A driver may set step > width*channels. Reshaping straight to (h, w, c)
    absorbs the padding into the image and shears it progressively down the
    frame -- subtle enough to look like a camera problem rather than a bug."""
    img = _gradient(h=8, w=10)
    pad = 7
    step = img.shape[1] * 3 + pad
    padded = np.zeros((img.shape[0], step), dtype=np.uint8)
    padded[:, : img.shape[1] * 3] = img.reshape(img.shape[0], -1)
    msg = FakeImage(
        height=img.shape[0], width=img.shape[1], encoding="bgr8",
        step=step, data=padded.tobytes(),
    )
    assert np.array_equal(to_bgr8(msg), img)


def test_step_of_zero_falls_back_to_dense_rows():
    img = _gradient()
    assert np.array_equal(to_bgr8(_msg(img, "bgr8", step=0)), img)


# ------------------------------------------------------------- writability
def test_result_is_writable_and_contiguous():
    """np.frombuffer yields a read-only view. ObjectDetector.draw and
    OpenVocabDetector.draw annotate in place, so a read-only frame raises inside
    OpenCV -- but only when publish_annotated is enabled, which is exactly the
    kind of bug that ships."""
    out = to_bgr8(_msg(_gradient(), "bgr8"))
    assert out.flags["WRITEABLE"]
    assert out.flags["C_CONTIGUOUS"]
    out[0, 0, 0] = 255  # must not raise


@pytest.mark.parametrize("encoding", ["rgb8", "mono8", "bgra8"])
def test_converted_results_are_writable_too(encoding):
    channels = {"rgb8": 3, "mono8": 1, "bgra8": 4}[encoding]
    img = _gradient(c=channels)
    if channels == 1:
        img = img.reshape(12, 16)
    out = to_bgr8(_msg(img, encoding))
    assert out.flags["WRITEABLE"]
    out[0, 0, 0] = 255


# ------------------------------------------------------------------ errors
def test_unknown_encoding_raises_rather_than_guessing():
    with pytest.raises(UnsupportedEncoding) as exc:
        to_bgr8(_msg(_gradient(), "16UC1"))
    assert "16uc1" in str(exc.value).lower()


def test_short_buffer_is_reported_not_reshaped():
    """A truncated frame must not become a confusing reshape traceback."""
    img = _gradient()
    msg = _msg(img, "bgr8")
    msg.data = msg.data[: len(msg.data) // 2]
    with pytest.raises(ValueError, match="short"):
        to_bgr8(msg)


# ------------------------------------------------------------------ encode
def test_fill_bgr8_sets_every_field_the_decoder_reads():
    img = _gradient()
    msg = fill_bgr8(FakeImage(), img)
    assert (msg.height, msg.width) == img.shape[:2]
    assert msg.encoding == "bgr8"
    assert msg.step == img.shape[1] * 3
    assert msg.is_bigendian == 0
    assert np.array_equal(to_bgr8(msg), img)


def test_fill_bgr8_leaves_the_header_alone():
    """Nodes reuse the *input* frame's header so capture time survives the hop;
    stamping a fresh one here would quietly destroy the Phase 6 measurement."""
    msg = FakeImage()
    msg.header = "sentinel"
    fill_bgr8(msg, _gradient())
    assert msg.header == "sentinel"


def test_fill_bgr8_round_trips_through_a_non_contiguous_view():
    """A cropped or flipped frame is a view with its own strides; tobytes() on
    it without ascontiguousarray would serialize the wrong pixels."""
    img = _gradient(h=12, w=16)
    view = img[:, ::-1]                       # mirrored, negative stride
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(to_bgr8(fill_bgr8(FakeImage(), view)), view)


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4), np.uint8),               # no channel axis
    np.zeros((4, 4, 4), np.uint8),            # BGRA, not BGR
    np.zeros((4, 4, 3), np.float32),          # wrong dtype
])
def test_fill_bgr8_rejects_arrays_it_cannot_represent(bad):
    with pytest.raises(ValueError):
        fill_bgr8(FakeImage(), bad)
