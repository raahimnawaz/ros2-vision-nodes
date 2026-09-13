#!/usr/bin/env python3
"""Phase 5 -- prove the two detector nodes really are a launch-time swap.

    source ros2/env.sh
    python3 ros2/tools/detector_swap.py
    python3 ros2/tools/detector_swap.py --queries "bus, person" --n 10

``detector_node`` (YOLO11, 80 fixed classes) and ``locate_node`` (OWLv2, open
vocabulary) claim to be interchangeable: same ``/detections`` topic, same
``vision_msgs/Detection2DArray``, same populated fields, so which one you launch
changes nothing downstream. That claim was never checked -- both nodes were
written to the same shape by hand, which is exactly the situation where two
implementations drift.

This runs each node in turn against the *same* frames and reports two things.

**The contract.** Which message fields each detector actually populates, and any
difference between them. A field one node fills and the other leaves at zero is
a downstream consumer that works with one detector and silently misbehaves with
the other -- the same class of bug as the display-coordinate mirror.

**The cost.** Model latency and node round-trip latency for both, measured
back-to-back on one machine in one process. The README has carried a YOLO number
(~30 fps) and an OWLv2 number (2.0 fps) that were measured at different times in
different ways, with a note admitting they were not comparable. This makes them
comparable.

Nodes are run sequentially, never together: they publish on the same topic by
design, so a downstream subscriber could not tell whose detections it received.
That is the point of the swap, and it is also why launching both at once is
explicitly warned against in the launch file.
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "ros2", "src", "gesture_bot_ros"))

from gesture_bot_ros.image_bridge import fill_bgr8  # noqa: E402


class FrameFeeder(Node):
    """Publishes one still frame on /image_raw on demand."""

    def __init__(self, frame):
        super().__init__("frame_feeder")
        self.frame = frame
        self._pub = self.create_publisher(Image, "image_raw", qos_profile_sensor_data)

    def send(self):
        msg = fill_bgr8(Image(), self.frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self._pub.publish(msg)
        return msg.header.stamp


class Sink(Node):
    """Collects whatever lands on /detections."""

    def __init__(self):
        super().__init__("detection_sink")
        self.received = []
        self.create_subscription(
            Detection2DArray, "detections", self.received.append, 10
        )


def populated_fields(array_msg):
    """Which parts of a Detection2DArray this detector actually filled in.

    Reported as a set of names rather than values, because the question is
    whether a downstream consumer can rely on a field being there -- not whether
    the two detectors found the same objects, which they will not.
    """
    fields = set()
    if array_msg.header.frame_id:
        fields.add("header.frame_id")
    if array_msg.header.stamp.sec or array_msg.header.stamp.nanosec:
        fields.add("header.stamp")
    for det in array_msg.detections:
        if det.id:
            fields.add("detection.id")
        if det.header.frame_id:
            fields.add("detection.header.frame_id")
        if det.bbox.size_x:
            fields.add("bbox.size_x")
        if det.bbox.size_y:
            fields.add("bbox.size_y")
        if det.bbox.center.position.x:
            fields.add("bbox.center.position.x")
        if det.bbox.center.position.y:
            fields.add("bbox.center.position.y")
        for res in det.results:
            if res.hypothesis.class_id:
                fields.add("results.hypothesis.class_id")
            if res.hypothesis.score:
                fields.add("results.hypothesis.score")
    return fields


def measure(label, make_node, frame, n, warmup):
    """Run one detector node against n frames. Returns a result dict."""
    print(f"\n=== {label} ===")
    t0 = time.perf_counter()
    node = make_node()
    load_s = time.perf_counter() - t0
    print(f"  load: {load_s:.1f}s")

    feeder = FrameFeeder(frame)
    sink = Sink()
    executor = SingleThreadedExecutor()
    for n_ in (node, feeder, sink):
        executor.add_node(n_)

    # Wait for the detector to actually subscribe and for the sink to be matched
    # to it. Publishing before discovery completes drops the frame silently, and
    # the first "latency" sample would really be a matchmaking measurement.
    deadline = time.time() + 10.0
    while time.time() < deadline and not (
        feeder.count_subscribers("image_raw") >= 1
        and sink.count_publishers("detections") >= 1
    ):
        executor.spin_once(timeout_sec=0.05)
    if feeder.count_subscribers("image_raw") < 1:
        print("  WARNING: detector never subscribed to /image_raw")

    # Model latency, straight from the detector -- no DDS in the number.
    model_ms = []
    for i in range(n + warmup):
        t = time.perf_counter()
        node.detector.detect(frame)
        dt = (time.perf_counter() - t) * 1e3
        if i >= warmup:
            model_ms.append(dt)

    # Node round-trip: publish a frame, spin until its detections come back.
    trip_ms = []
    last_seen = 0
    for _ in range(n):
        sink.received.clear()
        t = time.perf_counter()
        feeder.send()
        deadline = time.time() + 30.0
        while not sink.received and time.time() < deadline:
            executor.spin_once(timeout_sec=0.01)
        if sink.received:
            trip_ms.append((time.perf_counter() - t) * 1e3)
            last_seen = len(sink.received[-1].detections)

    sample = sink.received[-1] if sink.received else None
    fields = populated_fields(sample) if sample else set()

    for n_ in (node, feeder, sink):
        executor.remove_node(n_)
        n_.destroy_node()

    def stats(xs):
        if not xs:
            return "n/a"
        return (f"{statistics.median(xs):7.1f} ms median  "
                f"[{min(xs):.1f} - {max(xs):.1f}]  {1e3/statistics.median(xs):5.1f} fps")

    print(f"  model latency:      {stats(model_ms)}")
    print(f"  node round-trip:    {stats(trip_ms)}")
    print(f"  detections in last frame: {last_seen}")
    return {"label": label, "fields": fields, "model_ms": model_ms,
            "trip_ms": trip_ms, "detections": last_seen}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", default=os.path.join(_REPO, "testimg", "bus.jpg"))
    ap.add_argument("--queries", default="bus, person",
                    help="open-vocabulary queries for locate_node")
    ap.add_argument("--n", type=int, default=10, help="frames measured per detector")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--only", choices=["yolo", "owlv2"], default=None)
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")
    print(f"frame: {args.image}  {frame.shape[1]}x{frame.shape[0]}")

    # queries reach locate_node as a ROS parameter override, set the same way the
    # launch file sets it, rather than by reaching into the node afterwards --
    # this exercises the real parameter path. Applied at init so it is in place
    # before the node declares anything; detector_node has no `queries`
    # parameter and simply ignores the override.
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    init_args = [sys.argv[0], "--ros-args", "-p",
                 "queries:=[" + ",".join(f'"{q}"' for q in queries) + "]"]

    rclpy.init(args=init_args)
    results = []
    try:
        if args.only != "owlv2":
            from gesture_bot_ros.detector_node import DetectorNode
            results.append(measure("detector_node (YOLO11)", DetectorNode,
                                   frame, args.n, args.warmup))
        if args.only != "yolo":
            from gesture_bot_ros.locate_node import LocateNode
            results.append(measure("locate_node (OWLv2)", LocateNode,
                                   frame, args.n, args.warmup))
    finally:
        rclpy.shutdown()

    if len(results) == 2:
        a, b = results
        print("\n=== message contract ===")
        only_a = a["fields"] - b["fields"]
        only_b = b["fields"] - a["fields"]
        if not only_a and not only_b:
            print(f"  identical: both populate {len(a['fields'])} fields")
            for f in sorted(a["fields"]):
                print(f"    {f}")
        else:
            print("  MISMATCH -- the swap is not transparent:")
            for f in sorted(only_a):
                print(f"    only {a['label']}: {f}")
            for f in sorted(only_b):
                print(f"    only {b['label']}: {f}")

        print("\n=== back-to-back, same machine, same frames ===")
        am, bm = statistics.median(a["model_ms"]), statistics.median(b["model_ms"])
        print(f"  {a['label']:24s} {am:7.1f} ms  {1e3/am:5.1f} fps")
        print(f"  {b['label']:24s} {bm:7.1f} ms  {1e3/bm:5.1f} fps")
        print(f"  open vocabulary costs {bm/am:.1f}x the latency")


if __name__ == "__main__":
    main()
