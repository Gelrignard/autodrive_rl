#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Int32, Bool
from sensor_msgs.msg import Image, Imu, LaserScan, JointState
from geometry_msgs.msg import Vector3, Point
import numpy as np
import time


class RoboracerInterface(Node):
    def __init__(self):
        super().__init__('roboracer_interface')
        
        # Storage for received data
        self.data = {
            'reset_command': None,
            'best_lap_time': None,
            'collision_count': None,
            'front_camera': None,
            'imu_orientation': None,
            'imu_angular_velocity': None,
            'imu_linear_acceleration': None,
            'ips': None,  # 3D position
            'lap_count': None,
            'lap_time': None,
            'last_lap_time': None,
            'left_encoder': None,
            'lidar_ranges': None,  # 1080 range values
            'lidar_intensities': None,
            'right_encoder': None,
            'speed': None,
            'steering': None,
            'throttle': None
        }
        
        # Previous encoder values for velocity computation
        self.prev_encoder = {
            'left': None,
            'right': None,
            'timestamp': None
        }
        
        # Control parameters
        self.target_steering = 0.0
        self.target_throttle = 0.3
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers with appropriate QoS
        self.create_subscription(Bool, '/autodrive/reset_command', 
                                self.reset_command_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/best_lap_time', 
                                self.best_lap_time_callback, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', 
                                self.collision_count_callback, 10)
        self.create_subscription(Image, '/autodrive/roboracer_1/front_camera', 
                                self.front_camera_callback, qos_profile)
        self.create_subscription(Imu, '/autodrive/roboracer_1/imu', 
                                self.imu_callback, qos_profile)
        
        # Try Point instead of Vector3 for IPS
        self.create_subscription(Point, '/autodrive/roboracer_1/ips', 
                                self.ips_callback, qos_profile)
        
        self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', 
                                self.lap_count_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/lap_time', 
                                self.lap_time_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/last_lap_time', 
                                self.last_lap_time_callback, 10)
        self.create_subscription(JointState, '/autodrive/roboracer_1/left_encoder', 
                                self.left_encoder_callback, qos_profile)
        self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', 
                                self.lidar_callback, qos_profile)
        self.create_subscription(JointState, '/autodrive/roboracer_1/right_encoder', 
                                self.right_encoder_callback, qos_profile)
        self.create_subscription(Float32, '/autodrive/roboracer_1/speed', 
                                self.speed_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', 
                                self.steering_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/throttle', 
                                self.throttle_callback, 10)
        
        # Create publishers for commands
        self.reset_pub = self.create_publisher(Bool, '/autodrive/reset_command', 10)
        self.steering_cmd_pub = self.create_publisher(Float32, 
                                '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_cmd_pub = self.create_publisher(Float32, 
                                '/autodrive/roboracer_1/throttle_command', 10)
        
        # Create timer for periodic updates (1 Hz)
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        self.get_logger().info('Roboracer Interface Node Started')
        self.get_logger().info('Waiting for IPS data...')
    
    # Callback functions for subscribers
    def reset_command_callback(self, msg):
        self.data['reset_command'] = msg.data
    
    def best_lap_time_callback(self, msg):
        self.data['best_lap_time'] = msg.data
    
    def collision_count_callback(self, msg):
        self.data['collision_count'] = msg.data
    
    def front_camera_callback(self, msg):
        self.data['front_camera'] = f"Image: {msg.width}x{msg.height}"
    
    def imu_callback(self, msg):
        # Store all IMU components
        self.data['imu_orientation'] = [msg.orientation.x, msg.orientation.y, 
                                       msg.orientation.z, msg.orientation.w]
        self.data['imu_angular_velocity'] = [msg.angular_velocity.x, 
                                            msg.angular_velocity.y, 
                                            msg.angular_velocity.z]
        self.data['imu_linear_acceleration'] = [msg.linear_acceleration.x, 
                                               msg.linear_acceleration.y, 
                                               msg.linear_acceleration.z]
    
    def ips_callback(self, msg):
        # 3D position (x, y, z) - works for both Point and Vector3
        self.data['ips'] = [msg.x, msg.y, msg.z]
        # Only log first time to avoid spam
        if not hasattr(self, '_ips_received'):
            self.get_logger().info(f'IPS data received: x={msg.x:.2f}, y={msg.y:.2f}, z={msg.z:.2f}')
            self._ips_received = True
    
    def lap_count_callback(self, msg):
        self.data['lap_count'] = msg.data
    
    def lap_time_callback(self, msg):
        self.data['lap_time'] = msg.data
    
    def last_lap_time_callback(self, msg):
        self.data['last_lap_time'] = msg.data
    
    def left_encoder_callback(self, msg):
        # JointState message has position, velocity, and effort arrays
        if len(msg.position) > 0:
            current_time = time.time()
            position = msg.position[0]
            
            # Compute velocity from position change if not provided
            velocity = None
            if len(msg.velocity) > 0 and msg.velocity[0] != 0.0:
                velocity = msg.velocity[0]
            elif self.prev_encoder['left'] is not None and self.prev_encoder['timestamp'] is not None:
                dt = current_time - self.prev_encoder['timestamp']
                if dt > 0:
                    velocity = (position - self.prev_encoder['left']) / dt
            
            self.data['left_encoder'] = {
                'position': position,
                'velocity': velocity,
                'effort': msg.effort[0] if len(msg.effort) > 0 else None
            }
            
            # Update previous values
            self.prev_encoder['left'] = position
            self.prev_encoder['timestamp'] = current_time
    
    def lidar_callback(self, msg):
        # Store 1080 range values and intensities
        self.data['lidar_ranges'] = np.array(msg.ranges)
        self.data['lidar_intensities'] = np.array(msg.intensities)
    
    def right_encoder_callback(self, msg):
        # JointState message has position, velocity, and effort arrays
        if len(msg.position) > 0:
            current_time = time.time()
            position = msg.position[0]
            
            # Compute velocity from position change if not provided
            velocity = None
            if len(msg.velocity) > 0 and msg.velocity[0] != 0.0:
                velocity = msg.velocity[0]
            elif self.prev_encoder['right'] is not None and self.prev_encoder['timestamp'] is not None:
                dt = current_time - self.prev_encoder['timestamp']
                if dt > 0:
                    velocity = (position - self.prev_encoder['right']) / dt
            
            self.data['right_encoder'] = {
                'position': position,
                'velocity': velocity,
                'effort': msg.effort[0] if len(msg.effort) > 0 else None
            }
            
            # Update previous values
            self.prev_encoder['right'] = position
    
    def speed_callback(self, msg):
        self.data['speed'] = msg.data
    
    def steering_callback(self, msg):
        self.data['steering'] = msg.data
    
    def throttle_callback(self, msg):
        self.data['throttle'] = msg.data
    
    def get_lidar_ranges(self):
        """Get LiDAR ranges as numpy array (1080 values)"""
        return self.data.get('lidar_ranges')
    
    def get_lidar_front_ranges(self, num_rays=180):
        """Get front-facing LiDAR ranges (centered around index 540)"""
        ranges = self.data.get('lidar_ranges')
        if ranges is not None:
            center = len(ranges) // 2
            half_rays = num_rays // 2
            return ranges[center - half_rays:center + half_rays]
        return None
    
    def get_min_distance_ahead(self):
        """Get minimum distance in front of the vehicle"""
        front_ranges = self.get_lidar_front_ranges()
        if front_ranges is not None:
            # Filter out invalid readings (inf, nan)
            valid_ranges = front_ranges[np.isfinite(front_ranges)]
            if len(valid_ranges) > 0:
                return np.min(valid_ranges)
        return None
    
    def get_position(self):
        """Get 3D position from IPS"""
        return self.data.get('ips')
    
    def get_imu_data(self):
        """Get full IMU data"""
        return {
            'orientation': self.data.get('imu_orientation'),
            'angular_velocity': self.data.get('imu_angular_velocity'),
            'linear_acceleration': self.data.get('imu_linear_acceleration')
        }
    
    def get_encoder_data(self):
        """Get encoder data from both wheels"""
        return {
            'left': self.data.get('left_encoder'),
            'right': self.data.get('right_encoder')
        }
    
    # ADD YOUR CONTROL LOGIC HERE
    def compute_control(self):
        """
        Compute steering and throttle based on sensor data.
        Modify this function to implement your control algorithm.
        """
        # Example: Simple obstacle avoidance using LiDAR
        min_dist = self.get_min_distance_ahead()
        
        if min_dist is not None and min_dist < 2.0:
            # Obstacle ahead - slow down and turn
            self.target_throttle = 0.1
            self.target_steering = 0.3
        else:
            # Clear ahead - go straight
            self.target_throttle = 0.3
            self.target_steering = 0.0
        
        return self.target_steering, self.target_throttle
    
    # Timer callback - runs every second
    def timer_callback(self):
        # Compute control commands
        steering, throttle = self.compute_control()
        
        # Get min distance for display
        min_dist = self.get_min_distance_ahead()
        
        # Print received data
        print("\n" + "="*60)
        print("ROBORACER STATUS")
        print("="*60)
        print(f"Reset Command:     {self.data['reset_command']}")
        print(f"Best Lap Time:     {self.data['best_lap_time']}")
        print(f"Collision Count:   {self.data['collision_count']}")
        print(f"Front Camera:      {self.data['front_camera']}")
        
        # IMU data
        imu = self.get_imu_data()
        if imu['orientation']:
            print(f"IMU Orientation:   [{imu['orientation'][0]:.2f}, {imu['orientation'][1]:.2f}, "
                  f"{imu['orientation'][2]:.2f}, {imu['orientation'][3]:.2f}]")
        if imu['angular_velocity']:
            print(f"IMU Angular Vel:   [{imu['angular_velocity'][0]:.2f}, {imu['angular_velocity'][1]:.2f}, "
                  f"{imu['angular_velocity'][2]:.2f}]")
        if imu['linear_acceleration']:
            print(f"IMU Linear Accel:  [{imu['linear_acceleration'][0]:.2f}, {imu['linear_acceleration'][1]:.2f}, "
                  f"{imu['linear_acceleration'][2]:.2f}]")
        
        # Position (IPS)
        pos = self.get_position()
        if pos:
            print(f"Position (IPS):    [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        else:
            print(f"Position (IPS):    Not receiving data")
        
        print(f"Lap Count:         {self.data['lap_count']}")
        print(f"Lap Time:          {self.data['lap_time']:.2f}s" if self.data['lap_time'] else "Lap Time:          None")
        print(f"Last Lap Time:     {self.data['last_lap_time']}")
        
        # Encoder data
        encoders = self.get_encoder_data()
        if encoders['left'] is not None:
            vel_str = f"{encoders['left']['velocity']:.2f} rad/s" if encoders['left']['velocity'] is not None else "Computing..."
            print(f"Left Encoder:      Pos={encoders['left']['position']:.2f} rad, Vel={vel_str}")
        else:
            print(f"Left Encoder:      None")
        
        # LiDAR data
        lidar = self.data.get('lidar_ranges')
        if lidar is not None:
            if min_dist is not None:
                print(f"LiDAR:             {len(lidar)} ranges, Min ahead: {min_dist:.2f}m")
            else:
                print(f"LiDAR:             {len(lidar)} ranges, No valid data")
        else:
            print(f"LiDAR:             None")
        
        if encoders['right'] is not None:
            vel_str = f"{encoders['right']['velocity']:.2f} rad/s" if encoders['right']['velocity'] is not None else "Computing..."
            print(f"Right Encoder:     Pos={encoders['right']['position']:.2f} rad, Vel={vel_str}")
        else:
            print(f"Right Encoder:     None")
        
        print(f"Speed:             {self.data['speed']}")
        print(f"Steering:          {self.data['steering']}")
        print(f"Throttle:          {self.data['throttle']}")
        print("-"*60)
        print(f"COMMANDS: Steering={steering:.2f}, Throttle={throttle:.2f}")
        print("="*60)
        
        # Publish commands
        steering_msg = Float32()
        steering_msg.data = steering
        self.steering_cmd_pub.publish(steering_msg)
        
        throttle_msg = Float32()
        throttle_msg.data = throttle
        self.throttle_cmd_pub.publish(throttle_msg)
        
        self.get_logger().info(f'Published: Steering={steering:.2f}, Throttle={throttle:.2f}')
    
    def reset_simulation(self):
        """Call this to reset the simulation"""
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)
        self.get_logger().info('Sent reset command')


def main(args=None):
    rclpy.init(args=args)
    node = RoboracerInterface()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()