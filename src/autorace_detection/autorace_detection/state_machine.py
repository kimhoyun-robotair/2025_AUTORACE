#!/usr/bin/env python3

import enum
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class DriveState(enum.Enum):
    LANE_STAGE1 = 1
    COLOR_STAGE = 2
    LANE_STAGE2 = 3
    STOP_FINAL = 4
    DONE = 5


class LaneFSM(Node):
    """
    Three-phase mission with final stopline handling:
    1) Lane following
    2) Color-based straight driving
    3) Lane following
    4) Stop on stopline (after phases 1-3 complete)
    """

    def __init__(self) -> None:
        super().__init__("lane_fsm")

        # Color stability / grace parameters
        self.declare_parameter("color_min_duration_sec", 1.0)
        self.declare_parameter("color_unknown_grace_sec", 0.5)

        # Final stop handling
        self.declare_parameter("stop_wait_final_sec", 5.0)
        self.declare_parameter("reverse_scale", 5.0)

        self.state = DriveState.LANE_STAGE1
        self.color_stage_start: Optional[float] = None
        self.last_color_seen_time: Optional[float] = None

        self.last_lane_cmd: Optional[Twist] = None
        self.last_lane_valid: bool = False
        self.last_color_cmd: Optional[Twist] = None
        self.last_color: str = "unknown"
        self.last_stop_cmd: Optional[Twist] = None
        self.last_forward_speed: float = 0.0

        self.stop_detected = False
        self.stop_start_time: Optional[float] = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.current_speed_pub = self.create_publisher(Float32, "/fsm/current_speed", 10)

        self.create_subscription(Twist, "/lane/cmd_vel", self.lane_cmd_cb, 10)
        self.create_subscription(Bool, "/lane/has_lane", self.lane_valid_cb, 10)
        self.create_subscription(Twist, "/color/cmd_vel", self.color_cmd_cb, 10)
        self.create_subscription(String, "/lane/color", self.color_str_cb, 10)
        self.create_subscription(Bool, "/stopline/detected", self.stop_detect_cb, 10)
        self.create_subscription(Twist, "/stopline/reverse_cmd", self.stop_cmd_cb, 10)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Lane FSM node started (Lane -> Color -> Lane -> Stop).")

    def lane_cmd_cb(self, msg: Twist) -> None:
        self.last_lane_cmd = msg

    def lane_valid_cb(self, msg: Bool) -> None:
        self.last_lane_valid = msg.data

    def color_cmd_cb(self, msg: Twist) -> None:
        self.last_color_cmd = msg

    def color_str_cb(self, msg: String) -> None:
        self.last_color = msg.data.lower().strip()
        if self.state == DriveState.COLOR_STAGE and self.last_color in ("red", "blue"):
            self.last_color_seen_time = self._now()

    def stop_cmd_cb(self, msg: Twist) -> None:
        self.last_stop_cmd = msg

    def stop_detect_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        # Only honor stopline after lane+color+lane phases (i.e., when in LANE_STAGE2)
        if self.state == DriveState.LANE_STAGE2 and not self.stop_detected:
            self.stop_detected = True
            self.stop_start_time = self._now()
            self._set_state(DriveState.STOP_FINAL, reason="stopline detected after mission phases")

    def control_loop(self) -> None:
        twist = Twist()

        if self.state == DriveState.LANE_STAGE1:
            if self.last_lane_cmd and self.last_lane_valid:
                twist = self.last_lane_cmd
                self.last_forward_speed = abs(twist.linear.x)
            if self.last_color in ("red", "blue"):
                self.color_stage_start = self._now()
                self.last_color_seen_time = self._now()
                self._set_state(DriveState.COLOR_STAGE, reason=f"color={self.last_color}")
            self._publish(twist)
            return

        if self.state == DriveState.COLOR_STAGE:
            if self.last_color_cmd:
                twist = self.last_color_cmd
                self.last_forward_speed = abs(twist.linear.x)
            now = self._now()
            min_dur = float(self.get_parameter("color_min_duration_sec").value)
            grace = float(self.get_parameter("color_unknown_grace_sec").value)
            elapsed = now - (self.color_stage_start or now)
            time_since_last_color = (
                now - self.last_color_seen_time if self.last_color_seen_time is not None else grace
            )
            if self.last_color in ("red", "blue"):
                self.last_color_seen_time = now
            if elapsed >= min_dur and time_since_last_color >= grace and self.last_color == "unknown":
                self._set_state(DriveState.LANE_STAGE2, reason="color ended with grace")
            self._publish(twist)
            return

        if self.state == DriveState.LANE_STAGE2:
            if self.last_lane_cmd and self.last_lane_valid:
                twist = self.last_lane_cmd
                self.last_forward_speed = abs(twist.linear.x)
            self._publish(twist)
            return

        if self.state == DriveState.STOP_FINAL:
            twist = self._reverse_cmd()
            self._publish(twist, forward_speed=self.last_lane_cmd.linear.x)
            if self.stop_start_time is not None:
                elapsed = self._now() - self.stop_start_time
                if elapsed >= float(self.get_parameter("stop_wait_final_sec").value):
                    self._set_state(DriveState.DONE, reason="final stop complete")
                    self.get_logger().info("Final stop completed. Shutting down FSM node.")
                    rclpy.shutdown()
            return

        if self.state == DriveState.DONE:
            self._publish(Twist(), forward_speed=0.0)

    def _reverse_cmd(self) -> Twist:
        twist = Twist()
        if self.last_stop_cmd:
            twist = self.last_stop_cmd
        else:
            twist.linear.x = -abs(self.last_forward_speed) * float(
                self.get_parameter("reverse_scale").value
            )
        twist.angular.z = 0.0
        return twist

    def _publish(self, twist: Twist, forward_speed: Optional[float] = None) -> None:
        if forward_speed is None:
            forward_speed = abs(twist.linear.x)
        self.current_speed_pub.publish(Float32(data=float(forward_speed)))
        self.cmd_pub.publish(twist)

    def _set_state(self, new_state: DriveState, reason: str = "") -> None:
        if new_state == self.state:
            return
        self.get_logger().info(f"FSM state: {self.state.name} -> {new_state.name} ({reason})")
        self.state = new_state

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneFSM()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
