"""sensor_msgs/Image <-> BGR ndarray, without cv_bridge.

Why this exists
---------------
``cv_bridge`` is a compiled extension linked against the NumPy that shipped with
the ROS distro -- 1.26.4 on Jazzy. Every model in this repo now pulls NumPy 2:
``opencv-python`` 5.0.0.93 declares ``numpy>=2`` for Python 3.9+, so "just pin
numpy<2" is not available without giving up OpenCV. That put the perception
stack (torch, transformers, ultralytics, cv2) and the ROS stack (rclpy,
cv_bridge) on two mutually exclusive interpreters, and no node in ``ros2/``
could import both halves of what it needed.

The failure mode is the reason this module exists rather than a version pin.
Under NumPy 2, ``import cv_bridge`` **succeeds** -- it is
``cv_bridge.boost.cv_bridge_boost`` that is stale, and the ImportError it raises
is swallowed on the way up. The break only surfaces on the first real frame:

    >>> from cv_bridge import CvBridge          # fine
    >>> CvBridge().imgmsg_to_cv2(msg, "bgr8")   # KeyError: 16

An import check therefore reports a healthy bridge on a machine where every
frame will fail. That is the worst shape a dependency problem can take, and it
is worth removing rather than pinning around.

What is given up, and what is not
---------------------------------
cv_bridge converts between far more encodings than this does -- 16-bit depth
images, Bayer patterns, float32 disparity. This handles the 8-bit colour
encodings a webcam actually produces, and raises a named error listing what it
supports rather than returning wrong pixels for anything else. Nothing in this
repo has ever subscribed to a depth or Bayer topic.

What is gained beyond unblocking the import: no compiled dependency, so the
conversion is testable in CI with nothing installed but numpy and
opencv-python-headless -- see ``test/test_image_bridge.py``. Phase 7 replays a
bag through this path, and the conversion is the part most likely to break
silently.

Duck-typed on purpose: nothing here imports ``sensor_msgs``. ``to_bgr8`` reads
the five fields it needs off whatever it is handed, and ``fill_bgr8`` writes
into a message the caller constructed. That is what keeps the tests runnable
without ROS installed.
"""

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - only mono8/bgr8 work without it
    cv2 = None

#: Bytes per pixel for each encoding this module accepts.
CHANNELS = {
    "mono8": 1,
    "bgr8": 3,
    "rgb8": 3,
    "bgra8": 4,
    "rgba8": 4,
    "yuv422_yuy2": 2,
    "yuyv": 2,
}


class UnsupportedEncoding(ValueError):
    """Raised for an encoding this module will not guess at.

    Deliberately not a silent passthrough: reinterpreting an unknown encoding as
    BGR produces a plausible-looking image with wrong colours, which is far
    harder to notice than a raised error.
    """


def _needs_cv2(encoding):
    if cv2 is None:
        raise UnsupportedEncoding(
            f"converting {encoding!r} to bgr8 needs opencv-python, which is not "
            "installed"
        )


def to_bgr8(msg) -> np.ndarray:
    """A ``sensor_msgs/Image`` as a writable, contiguous BGR uint8 array.

    Writable and contiguous is not incidental. ``np.frombuffer`` returns a
    read-only view of the message buffer, and both ``ObjectDetector.draw`` and
    ``OpenVocabDetector.draw`` annotate the frame **in place** -- handing them a
    read-only array raises inside OpenCV on the first box drawn, only when
    ``publish_annotated`` happens to be on.
    """
    encoding = (msg.encoding or "").lower()
    if encoding not in CHANNELS:
        raise UnsupportedEncoding(
            f"encoding {encoding!r} is not supported; expected one of "
            f"{sorted(CHANNELS)}"
        )

    channels = CHANNELS[encoding]
    height, width = int(msg.height), int(msg.width)
    # step is the row stride in bytes and may exceed width*channels: drivers are
    # allowed to pad rows for alignment. Reshaping straight to (h, w, c) would
    # silently shear the image on any camera that does.
    step = int(msg.step) or width * channels

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if buf.size < expected:
        raise ValueError(
            f"image data is short: got {buf.size} bytes, need {expected} "
            f"({height} rows x {step} step)"
        )
    rows = buf[:expected].reshape(height, step)
    pixels = rows[:, : width * channels].reshape(height, width, channels)

    if encoding == "bgr8":
        # .copy(), NOT np.ascontiguousarray(): the slice above is already
        # C-contiguous whenever step has no padding, so ascontiguousarray
        # returns it unchanged -- still a read-only view of the message buffer.
        # copy() is the only one of the two that guarantees a writable array.
        return pixels.copy()

    _needs_cv2(encoding)
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    # yuv422_yuy2 / yuyv -- v4l2_camera emits this when output_encoding is left
    # at the camera's native format instead of being transcoded to rgb8.
    return cv2.cvtColor(pixels, cv2.COLOR_YUV2BGR_YUY2)


def fill_bgr8(msg, img: np.ndarray):
    """Fill an already-constructed ``sensor_msgs/Image`` from a BGR array.

    The caller constructs the message so this module never imports
    ``sensor_msgs``; it also leaves ``header`` alone, because the header a node
    wants is almost always the *input* frame's, not a fresh one -- carrying the
    capture stamp through is what makes end-to-end latency measurable at all
    (Phase 6).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR array, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {img.dtype}")

    img = np.ascontiguousarray(img)
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(img.shape[1] * 3)
    msg.data = img.tobytes()
    return msg
