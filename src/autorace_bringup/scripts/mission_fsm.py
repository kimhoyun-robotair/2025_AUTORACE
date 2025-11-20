#!/usr/bin/env python3
"""
Mission FSM (simplified sequence):
  1) LANE_1: follow lane (lane_following -> /control/lane_cmd)
  2) STOP_1: stop-line detected -> hold stop for 2s
  3) COLOR: color zone drive (color_based_driving -> /control/color_cmd) while red/blue
  4) LANE_2: fall back to lane following when color disappeared
  5) STOP_2: next stop-line -> hold stop for 5s
  6) DONE: stay stopped

Inputs:
  - /control/lane_cmd (Twist)          : from lane_following.py
  - /control/color_cmd (Twist)         : from color_based_driving.py
  - /detections/stopline (Bool)        : from stop_line_stop.py
  - /lane/color (String)               : from color_detection.py ("red"/"blue"/"unknown")

Output:
  - /cmd_vel (Twist)                   : selected command or zero during stops/DONE
  - /mission/state (String)            : current state name (for debugging)

Note: control nodes now publish to dedicated channels; this FSM is the only publisher to /cmd_vel.
"""

import enum
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


class Phase(enum.Enum):
    LANE_1 = 1
    STOP_1 = 2
    COLOR = 3
    LANE_2 = 4
    STOP_2 = 5
    DONE = 6
    FAILSAFE = 99


class MissionFSM(Node):
    def __init__(self):
        super().__init__("mission_fsm")

        # State & caches
        self.phase = Phase.LANE_1
        self.current_color = "unknown"
        self.stop_detected = False
        self.stop_hold_until_ns: Optional[int] = None

        # Command buffers
        self.cmd_lane: Twist = Twist()
        self.cmd_color: Twist = Twist()

        # Subscribers
        self.create_subscription(Twist, "/control/lane_cmd", self.on_lane_cmd, 10)
        self.create_subscription(Twist, "/control/color_cmd", self.on_color_cmd, 10)
        self.create_subscription(Bool, "/detections/stopline", self.on_stopline, 10)
        self.create_subscription(String, "/lane/color", self.on_color_detect, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_state = self.create_publisher(String, "/mission/state", 10)

        # Timers
        self.timer = self.create_timer(0.05, self.tick)  # 20 Hz tick

        self.get_logger().info("MissionFSM ready: lane -> stop(2s) -> color -> lane -> stop(5s) -> done")

    # ---- Callbacks ----
    def on_lane_cmd(self, msg: Twist):
        self.cmd_lane = msg

    def on_color_cmd(self, msg: Twist):
        self.cmd_color = msg

    def on_stopline(self, msg: Bool):
        self.stop_detected = msg.data

    def on_color_detect(self, msg: String):
        self.current_color = msg.data.lower().strip()

    # ---- Helper ----
    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def holding(self) -> bool:
        return self.stop_hold_until_ns is not None and self.now_ns() < self.stop_hold_until_ns

    def tick(self):
        prev = self.phase

        # State transitions
        if self.phase == Phase.LANE_1:
            if self.stop_detected:
                self.phase = Phase.STOP_1
                self.stop_hold_until_ns = self.now_ns() + int(2.0 * 1e9)
        elif self.phase == Phase.STOP_1:
            if self.holding():
                pass
            elif self.current_color in ("red", "blue"):
                self.phase = Phase.COLOR
            else:
                # if color not ready, stay stopped but don't re-trigger timer
                pass
        elif self.phase == Phase.COLOR:
            if self.current_color not in ("red", "blue"):
                self.phase = Phase.LANE_2
        elif self.phase == Phase.LANE_2:
            if self.stop_detected:
                self.phase = Phase.STOP_2
                self.stop_hold_until_ns = self.now_ns() + int(5.0 * 1e9)
        elif self.phase == Phase.STOP_2:
            if not self.holding():
                self.phase = Phase.DONE
        elif self.phase == Phase.DONE:
            pass
        elif self.phase == Phase.FAILSAFE:
            pass

        # Command selection
        if self.phase in (Phase.STOP_1, Phase.STOP_2, Phase.DONE, Phase.FAILSAFE) or self.stop_detected:
            # stop_detected guard prevents brief overlap before timer starts
            cmd_out = Twist()
            cmd_out.linear.x = -0.32  # brake backward; angular stays 0
        elif self.phase in (Phase.LANE_1, Phase.LANE_2):
            cmd_out = self.cmd_lane
        elif self.phase == Phase.COLOR:
            cmd_out = self.cmd_color
        else:
            cmd_out = Twist()

        self.pub_cmd.publish(cmd_out)

        if self.phase != prev:
            self.get_logger().info(f"State: {prev.name} -> {self.phase.name}")
        self.pub_state.publish(String(data=self.phase.name))


def main(args=None):
    rclpy.init(args=args)
    node = MissionFSM()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
