"""Locate and import the framework-free gesture_bot modules.

The ROS2 layer deliberately contains no control logic. ``decision.py``,
``actuators.py`` and ``perception.py`` are imported *unchanged* from the main
repo -- which is the whole reason ``decision.py`` was written with no cv2 or
mediapipe import. The nodes here only handle transport, parameters and
failsafes.

Because those modules live in the repo rather than in this ament package, the
source tree has to be found at runtime. Resolution order:

1. ``$GESTURE_BOT_SRC`` -- path to the repo root (the directory *containing*
   ``gesture_bot/``). Set this in your launch file or shell.
2. ``~/vision_demos`` -- the default checkout location used by the repo README.

Relative-to-``__file__`` resolution is deliberately not attempted: after
``colcon build`` this module lives under ``install/`` and no longer sits near
the repo.
"""

import os
import sys

_CANDIDATE_ENV = "GESTURE_BOT_SRC"
_DEFAULT = os.path.expanduser("~/vision_demos")


def _repo_root() -> str:
    tried = []
    for root in (os.environ.get(_CANDIDATE_ENV), _DEFAULT):
        if not root:
            continue
        root = os.path.abspath(os.path.expanduser(root))
        tried.append(root)
        if os.path.isfile(os.path.join(root, "gesture_bot", "decision.py")):
            return root
    raise ImportError(
        "Could not find the gesture_bot source tree. Set "
        f"{_CANDIDATE_ENV}=/path/to/vision_demos (the directory containing "
        f"gesture_bot/). Tried: {tried or '(nothing)'}"
    )


def ensure_on_path() -> str:
    """Put the repo's ``gesture_bot/`` package on sys.path. Returns the repo root."""
    root = _repo_root()
    for entry in (root, os.path.join(root, "gesture_bot")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root


ensure_on_path()

# Imported after the path is set up. `decision` is dependency-free; `actuators`
# needs cv2 + numpy; `perception` additionally needs mediapipe and the
# downloaded .task bundle, so it is imported lazily by gesture_node only.
from decision import GESTURE_MODES, ControllerConfig, GestureController

# ControllerConfig fields that decision_node exposes as live-tunable ROS
# parameters. Kept here rather than in the node so it can be checked against
# the dataclass without importing rclpy -- see test/test_core.py. A field
# renamed in decision.py without updating this tuple would otherwise silently
# stop being tunable.
TUNABLE = (
    "v_forward",
    "v_fast",
    "v_reverse",
    "w_rotate",
    "steer_gain",
    "confidence_min",
    "stable_frames",
    "lost_frames",
)

# Counts, not physical quantities: declared as ROS integer parameters.
INTEGER_PARAMS = ("stable_frames", "lost_frames")

# Node parameters that are not ControllerConfig fields but still need bounds.
POSITIVE_PARAMS = ("publish_rate", "gesture_timeout")

# Score thresholds. Semantically the same check as confidence_min, but these are
# detector parameters rather than ControllerConfig fields.
UNIT_INTERVAL_PARAMS = ("confidence_min", "conf")


def normalize_queries(value):
    """Clean an open-vocabulary query list, or raise ValueError.

    Whitespace-only entries are dropped rather than passed to the tokenizer,
    where they become a valid-but-meaningless query that quietly matches
    everything. A list that is empty after cleaning is an error, not an empty
    vocabulary: OWLv2 with no queries returns no detections forever, and the
    node would look merely blind rather than misconfigured.
    """
    if isinstance(value, (str, bytes)):
        # A bare string would otherwise iterate character by character and
        # produce one query per letter.
        raise ValueError("queries must be a list of strings, not a single string")
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"queries must be a list of strings, got {type(value).__name__}")
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"queries entries must be strings, got {item!r}")
        item = item.strip()
        if item:
            cleaned.append(item)
    if not cleaned:
        raise ValueError("queries must contain at least one non-empty string")
    return cleaned


def reject_parameter(name, value):
    """Why setting ``name`` to ``value`` would be unsafe, or None if it is fine.

    Lives here rather than on the node for the same reason ``TUNABLE`` does: it
    can be tested without rclpy. See ``test/test_core.py``.
    """
    try:
        if name in INTEGER_PARAMS:
            if int(value) < 1:
                return f"{name} must be >= 1"
        elif name == "queries":
            try:
                normalize_queries(value)
            except ValueError as exc:
                return str(exc)
        elif name in UNIT_INTERVAL_PARAMS:
            if not 0.0 <= float(value) <= 1.0:
                return f"{name} must be in [0, 1]"
        elif name in POSITIVE_PARAMS:
            if float(value) <= 0.0:
                return f"{name} must be > 0"
        elif name == "steer_gain":
            # Negative gain inverts steering. That is exactly the bug the
            # display-coordinate contract exists to prevent, so do not let a
            # live `ros2 param set` reintroduce it by hand.
            if float(value) < 0.0:
                return "steer_gain must be >= 0 (negative inverts steering)"
    except (TypeError, ValueError):
        return f"{name} got a value of the wrong type: {value!r}"
    return None


def reject_batch(items):
    """First rejection reason across an iterable of ``(name, value)``, else None.

    rclpy applies *nothing* when a set-parameters callback returns unsuccessful,
    so the whole batch has to be validated before any of it is applied. Applying
    as you iterate leaves the node holding values the parameter store rejected,
    and ``ros2 param get`` then disagrees with how the robot is behaving.
    """
    for name, value in items:
        reason = reject_parameter(name, value)
        if reason is not None:
            return reason
    return None


__all__ = [
    "GESTURE_MODES",
    "INTEGER_PARAMS",
    "POSITIVE_PARAMS",
    "TUNABLE",
    "UNIT_INTERVAL_PARAMS",
    "ControllerConfig",
    "GestureController",
    "ensure_on_path",
    "normalize_queries",
    "reject_batch",
    "reject_parameter",
]
