#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import serial
import math
import sys
import time

# Helper for quaternion conversion (removes tf dependency)
def quaternion_from_euler(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

class RazorImuNode(Node):
    def __init__(self):
        super().__init__('razor_node')

        # 1. Declare Parameters (기존 get_param 대체)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('port', '/dev/ttyUSB0'),
                ('topic', 'imu'),
                ('frame_id', 'base_imu_link'),
                ('accel_x_min', -250.0), ('accel_x_max', 250.0),
                ('accel_y_min', -250.0), ('accel_y_max', 250.0),
                ('accel_z_min', -250.0), ('accel_z_max', 250.0),
                ('magn_x_min', -600.0), ('magn_x_max', 600.0),
                ('magn_y_min', -600.0), ('magn_y_max', 600.0),
                ('magn_z_min', -600.0), ('magn_z_max', 600.0),
                ('calibration_magn_use_extended', False),
                ('magn_ellipsoid_center', [0.0, 0.0, 0.0]),
                ('magn_ellipsoid_transform', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), # Flattened 3x3
                ('imu_yaw_calibration', 0.0),
                ('gyro_average_offset_x', 0.0),
                ('gyro_average_offset_y', 0.0),
                ('gyro_average_offset_z', 0.0),
            ]
        )

        # 2. Read Parameters
        self.port = self.get_parameter('port').value
        topic_name = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.imu_yaw_calibration = self.get_parameter('imu_yaw_calibration').value

        # 3. Setup Serial
        self.get_logger().info(f"Opening {self.port}...")
        try:
            self.ser = serial.Serial(port=self.port, baudrate=57600, timeout=1)
        except serial.serialutil.SerialException:
            self.get_logger().error(f"IMU not found at port {self.port}")
            sys.exit(2)

        # 4. Setup Publishers
        self.pub_imu = self.create_publisher(Imu, topic_name, 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, 'diagnostics', 10)

        # 5. Initialize Variables
        self.seq = 0
        self.accel_factor = 9.806 / 256.0
        self.degrees2rad = math.pi / 180.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # 6. Configure Hardware (기존 코드의 보정값 전송 로직)
        self.configure_board()

        # 7. Setup Timer (Main Loop) - 50Hz
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.diag_timer = self.create_timer(1.0, self.diag_callback) # 1Hz diagnostics

        # 8. Setup Parameter Callback (Dynamic Reconfigure 대체)
        self.add_on_set_parameters_callback(self.parameters_callback)

    def configure_board(self):
        self.get_logger().info("Giving the razor IMU board 5 seconds to boot...")
        time.sleep(5.0)

        # Stop datastream
        self.ser.write(('#o0').encode("utf-8"))
        self.ser.readline() # discard

        # Set output mode
        self.ser.write(('#ox').encode("utf-8"))

        self.get_logger().info("Writing calibration values...")
        
        def get_p(name): return self.get_parameter(name).value

        # Accelerometer
        self.ser.write(f"#caxm{get_p('accel_x_min')}".encode("utf-8"))
        self.ser.write(f"#caxM{get_p('accel_x_max')}".encode("utf-8"))
        self.ser.write(f"#caym{get_p('accel_y_min')}".encode("utf-8"))
        self.ser.write(f"#cayM{get_p('accel_y_max')}".encode("utf-8"))
        self.ser.write(f"#cazm{get_p('accel_z_min')}".encode("utf-8"))
        self.ser.write(f"#cazM{get_p('accel_z_max')}".encode("utf-8"))

        # Magnetometer
        if not get_p('calibration_magn_use_extended'):
            self.ser.write(f"#cmxm{get_p('magn_x_min')}".encode("utf-8"))
            self.ser.write(f"#cmxM{get_p('magn_x_max')}".encode("utf-8"))
            self.ser.write(f"#cmym{get_p('magn_y_min')}".encode("utf-8"))
            self.ser.write(f"#cmyM{get_p('magn_y_max')}".encode("utf-8"))
            self.ser.write(f"#cmzm{get_p('magn_z_min')}".encode("utf-8"))
            self.ser.write(f"#cmzM{get_p('magn_z_max')}".encode("utf-8"))
        else:
            c = get_p('magn_ellipsoid_center')
            t = get_p('magn_ellipsoid_transform') # flattened list expected from yaml
            # Reconstruct if needed or use flat indexing
            # Note: ROS2 yaml lists are flat. Assuming 3x3 is passed as 9 elements
            self.ser.write(f"#ccx{c[0]}".encode("utf-8"))
            self.ser.write(f"#ccy{c[1]}".encode("utf-8"))
            self.ser.write(f"#ccz{c[2]}".encode("utf-8"))
            self.ser.write(f"#ctxX{t[0]}".encode("utf-8"))
            self.ser.write(f"#ctxY{t[1]}".encode("utf-8"))
            self.ser.write(f"#ctxZ{t[2]}".encode("utf-8"))
            self.ser.write(f"#ctyX{t[3]}".encode("utf-8"))
            self.ser.write(f"#ctyY{t[4]}".encode("utf-8"))
            self.ser.write(f"#ctyZ{t[5]}".encode("utf-8"))
            self.ser.write(f"#ctzX{t[6]}".encode("utf-8"))
            self.ser.write(f"#ctzY{t[7]}".encode("utf-8"))
            self.ser.write(f"#ctzZ{t[8]}".encode("utf-8"))

        # Gyro
        self.ser.write(f"#cgx{get_p('gyro_average_offset_x')}".encode("utf-8"))
        self.ser.write(f"#cgy{get_p('gyro_average_offset_y')}".encode("utf-8"))
        self.ser.write(f"#cgz{get_p('gyro_average_offset_z')}".encode("utf-8"))

        # Verify
        self.ser.flushInput()
        self.ser.write(('#p').encode("utf-8"))
        calib_data = self.ser.readline().decode('utf-8', errors='replace')
        self.get_logger().info(f"Calibration loaded: {calib_data.strip()}")

        # Start datastream
        self.ser.write(('#o1').encode("utf-8"))
        self.get_logger().info("Flushing first 200 IMU entries...")
        for _ in range(200):
            self.ser.readline()
        self.get_logger().info("Publishing IMU data...")

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'imu_yaw_calibration':
                self.imu_yaw_calibration = param.value
                self.get_logger().info(f"Reconfigure request for yaw_calibration: {self.imu_yaw_calibration}")
        return SetParametersResult(successful=True)

    def timer_callback(self):
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode("utf-8", errors='replace').strip()
            if "#YPRAG=" in line:
                self.parse_and_publish(line)

    def parse_and_publish(self, line):
        line = line.replace("#YPRAG=", "")
        words = line.split(",")
        if len(words) != 9:
            self.get_logger().warn("Bad IMU data or bad sync")
            return

        try:
            # Parsing Logic from original code
            yaw_deg = -float(words[0])
            yaw_deg += self.imu_yaw_calibration
            if yaw_deg > 180.0: yaw_deg -= 360.0
            if yaw_deg < -180.0: yaw_deg += 360.0
            
            self.yaw = yaw_deg * self.degrees2rad
            self.pitch = -float(words[1]) * self.degrees2rad
            self.roll = float(words[2]) * self.degrees2rad

            # Populate Imu Message
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = self.frame_id
            # imu_msg.header.seq is deprecated in ROS 2
            
            q = quaternion_from_euler(self.roll, self.pitch, self.yaw)
            imu_msg.orientation.x = q[0]
            imu_msg.orientation.y = q[1]
            imu_msg.orientation.z = q[2]
            imu_msg.orientation.w = q[3]
            
            # Coordinate conversions per original code
            imu_msg.linear_acceleration.x = -float(words[3]) * self.accel_factor
            imu_msg.linear_acceleration.y = float(words[4]) * self.accel_factor
            imu_msg.linear_acceleration.z = float(words[5]) * self.accel_factor
            
            imu_msg.angular_velocity.x = float(words[6])
            imu_msg.angular_velocity.y = -float(words[7])
            imu_msg.angular_velocity.z = -float(words[8])

            # Covariances
            imu_msg.orientation_covariance = [0.0025, 0.0, 0.0, 0.0, 0.0025, 0.0, 0.0, 0.0, 0.0025]
            imu_msg.angular_velocity_covariance = [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02]
            imu_msg.linear_acceleration_covariance = [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04]

            self.pub_imu.publish(imu_msg)
            self.seq += 1

        except ValueError as e:
            self.get_logger().warn(f"Parsing error: {e}")

    def diag_callback(self):
        diag_arr = DiagnosticArray()
        diag_arr.header.stamp = self.get_clock().now().to_msg()
        diag_arr.header.frame_id = '1'
        
        diag_msg = DiagnosticStatus()
        diag_msg.name = 'Razor_Imu'
        diag_msg.level = DiagnosticStatus.OK
        diag_msg.message = 'Received AHRS measurement'
        
        diag_msg.values.append(KeyValue(key='roll (deg)', value=str(self.roll * (180.0 / math.pi))))
        diag_msg.values.append(KeyValue(key='pitch (deg)', value=str(self.pitch * (180.0 / math.pi))))
        diag_msg.values.append(KeyValue(key='yaw (deg)', value=str(self.yaw * (180.0 / math.pi))))
        diag_msg.values.append(KeyValue(key='sequence number', value=str(self.seq)))
        
        diag_arr.status.append(diag_msg)
        self.pub_diag.publish(diag_arr)

def main(args=None):
    rclpy.init(args=args)
    node = RazorImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
