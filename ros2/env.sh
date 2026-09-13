# Source this before building or launching anything in ros2/.
#
#     source ros2/env.sh
#     ros2 launch gesture_bot_ros gesture_bot.launch.py
#
# Why a script rather than a line in the README
# ---------------------------------------------
# The nodes need two package sets that are installed in different places and
# cannot be merged by installing one into the other:
#
#   * ROS Jazzy's python packages (rclpy, sensor_msgs, vision_msgs, rosbag2_py)
#     live in /opt/ros/jazzy and are built for the system interpreter.
#   * The models (torch, transformers, ultralytics, mediapipe, opencv) live in a
#     venv, because the versions this repo needs are not the versions apt ships.
#
# A venv with --system-site-packages does not solve it either: that puts the
# venv's NumPy 2 ahead of the distro's 1.26, which is fine for everything except
# compiled extensions built against 1.26. cv_bridge was the only one, which is
# why it is no longer a dependency -- see gesture_bot_ros/image_bridge.py.
#
# So: system interpreter, ROS on the path, venv site-packages prepended. Verified
# end to end -- a real DDS hop carrying an Image, decoded by image_bridge, with a
# torch CUDA op in the same process.
#
# Override any of these before sourcing if your layout differs.

: "${ROS_SETUP:=/opt/ros/jazzy/setup.bash}"
: "${VISION_DEMOS_VENV:=$HOME/vision_demos_env}"

# Repo root -- the directory containing gesture_bot/. _core.py reads this.
if [ -z "${GESTURE_BOT_SRC:-}" ]; then
    _env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    GESTURE_BOT_SRC="$(cd "$_env_dir/.." && pwd)"
    export GESTURE_BOT_SRC
    unset _env_dir
fi

# shellcheck disable=SC1090
. "$ROS_SETUP"

# The workspace overlay, if it has been built.
if [ -f "$GESTURE_BOT_SRC/ros2/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    . "$GESTURE_BOT_SRC/ros2/install/setup.bash"
fi

_venv_sp="$VISION_DEMOS_VENV/lib/python3.12/site-packages"
if [ -d "$_venv_sp" ]; then
    # Prepended, so the venv's numpy/opencv/torch win over the distro's.
    export PYTHONPATH="$_venv_sp${PYTHONPATH:+:$PYTHONPATH}"
else
    echo "ros2/env.sh: warning: no venv site-packages at $_venv_sp" >&2
    echo "  set VISION_DEMOS_VENV=/path/to/venv before sourcing" >&2
fi
unset _venv_sp

# NOT activating the venv: `python3` must stay the system interpreter, because
# that is the one rclpy's compiled extension was built for. Putting the venv's
# bin/ on PATH would shadow it and break every node with a bare ImportError.
