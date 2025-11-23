#!/usr/bin/env python3

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from builtin_interfaces.msg import Time as RosTime
from std_srvs.srv import SetBool, Trigger

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def stamp_to_str(stamp: RosTime) -> str:
    tsec = int(stamp.sec)
    tnano = int(stamp.nanosec)
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(tsec)) + f"_{tnano:09d}"


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__("image_saver")

        # ---- Parameters ----
        self.declare_parameter("input_image_topic", "/camera/image_raw")
        self.declare_parameter("input_msg_type", "image")   # "image" or "compressed"
        self.declare_parameter("output_dir", "yolo_raw/images")
        self.declare_parameter("image_format", "jpg")       # "jpg" or "png"
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("png_compression", 3)
        self.declare_parameter("prefix", "")
        self.declare_parameter("use_stamp_subdir", False)
        self.declare_parameter("enabled", True)
        self.declare_parameter("save_interval_sec", 0.5)    # 핵심: N초마다 1장. 0이면 모든 프레임

        # ---- Read params ----
        self.input_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.msg_type = self.get_parameter("input_msg_type").get_parameter_value().string_value.lower()
        self.out_dir = Path(self.get_parameter("output_dir").get_parameter_value().string_value)
        self.img_fmt = self.get_parameter("image_format").get_parameter_value().string_value.lower()
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").get_parameter_value().integer_value)
        self.png_compression = int(self.get_parameter("png_compression").get_parameter_value().integer_value)
        self.prefix = self.get_parameter("prefix").get_parameter_value().string_value
        self.use_stamp_subdir = bool(self.get_parameter("use_stamp_subdir").get_parameter_value().bool_value)
        self.enabled = bool(self.get_parameter("enabled").get_parameter_value().bool_value)
        self.save_interval = float(self.get_parameter("save_interval_sec").get_parameter_value().double_value)

        if self.img_fmt not in ("jpg", "jpeg", "png"):
            self.get_logger().warn(f"Unsupported image_format '{self.img_fmt}', fallback to 'jpg'")
            self.img_fmt = "jpg"

        ensure_dir(self.out_dir)

        # QoS: sensor data
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.bridge = CvBridge()
        self.last_save_walltime: Optional[float] = None

        # Subscriber
        if self.msg_type == "compressed":
            self.sub = self.create_subscription(CompressedImage, self.input_topic, self.compressed_cb, qos)
            self.get_logger().info(f"Subscribed to CompressedImage: {self.input_topic}")
        else:
            self.sub = self.create_subscription(Image, self.input_topic, self.image_cb, qos)
            self.get_logger().info(f"Subscribed to Image: {self.input_topic}")

        # Services
        self.srv_toggle = self.create_service(SetBool, "capture_toggle", self.on_toggle)
        self.srv_snapshot = self.create_service(Trigger, "snapshot", self.on_snapshot)

        self.get_logger().info(
            f"Output: {self.out_dir} | fmt: {self.img_fmt} | interval: {self.save_interval}s | enabled={self.enabled}"
        )

        self._latest_cvimg = None
        self._latest_stamp = None

    # --------- Callbacks ---------
    def image_cb(self, msg: Image):
        try:
            cvimg = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return
        self._latest_cvimg = cvimg
        self._latest_stamp = msg.header.stamp
        self._maybe_save(cvimg, msg.header.stamp)

    def compressed_cb(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cvimg = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Compressed decode failed: {e}")
            return
        self._latest_cvimg = cvimg
        self._latest_stamp = msg.header.stamp
        self._maybe_save(cvimg, msg.header.stamp)

    # --------- Services ---------
    def on_toggle(self, req: SetBool.Request, res: SetBool.Response):
        self.enabled = bool(req.data)
        res.success = True
        res.message = f"capture {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(res.message)
        return res

    def on_snapshot(self, req: Trigger.Request, res: Trigger.Response):
        if self._latest_cvimg is not None and self._latest_stamp is not None:
            try:
                filepath = self._save_image(self._latest_cvimg, self._latest_stamp)
                res.success = True
                res.message = f"snapshot saved: {filepath}"
                self.get_logger().info(res.message)
                # 스냅샷은 간격과 무관하게 저장했으므로 last_save_walltime 갱신
                self.last_save_walltime = time.time()
                return res
            except Exception as e:
                res.success = False
                res.message = f"snapshot failed: {e}"
                self.get_logger().error(res.message)
                return res
        else:
            res.success = False
            res.message = "no image received yet"
            return res

    # --------- Internals ---------
    def _maybe_save(self, cvimg, stamp: RosTime):
        if not self.enabled:
            return

        now = time.time()
        if self.save_interval > 0.0 and self.last_save_walltime is not None:
            if (now - self.last_save_walltime) < self.save_interval:
                return

        # 저장
        self._save_image(cvimg, stamp)
        self.last_save_walltime = now

    def _save_image(self, cvimg, stamp: RosTime) -> str:
        out_dir = self.out_dir
        if self.use_stamp_subdir:
            date_str = time.strftime("%Y-%m-%d", time.localtime(int(stamp.sec)))
            out_dir = self.out_dir / date_str
            ensure_dir(out_dir)

        name_core = stamp_to_str(stamp)
        if self.prefix:
            name_core = f"{self.prefix}_{name_core}"

        ext = "jpg" if self.img_fmt in ("jpg", "jpeg") else "png"
        path = out_dir / f"{name_core}.{ext}"

        if ext == "jpg":
            ok, buf = cv2.imencode(".jpg", cvimg, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        else:
            ok, buf = cv2.imencode(".png", cvimg, [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.png_compression)])

        if not ok:
            raise RuntimeError("cv2.imencode failed")

        with open(path, "wb") as f:
            f.write(buf.tobytes())

        self.get_logger().debug(f"Saved {path}")
        return str(path)


def main():
    rclpy.init()
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
