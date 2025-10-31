#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Int32, Bool
from sensor_msgs.msg import Image, Imu, LaserScan, JointState
from geometry_msgs.msg import Point
import time
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import math
import wandb
from wandb.integration.sb3 import WandbCallback


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw)"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(np.clip(sinp, -1.0, 1.0))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


class RoboracerEnv(gym.Env, Node):
    """Custom Gym Environment for Roboracer using ROS 2"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_steps=1000):
        # Initialize ROS 2 node first
        if not rclpy.ok():
            rclpy.init()
        Node.__init__(self, 'roboracer_ppo_env')
        
        # Initialize Gym environment
        gym.Env.__init__(self)
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # Observation space: 
        # - 36 LiDAR readings (sparse)
        # - pose_x, pose_y, pose_theta
        # - steering_angle
        # - speed
        # - angular_velocity_z
        # - left_encoder_velocity
        # - right_encoder_velocity
        # - collision_count (normalized)
        # - lap_time (normalized)
        # - lap_count
        # Total: 47 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(47,),
            dtype=np.float32
        )
        
        # Action space: [steering, throttle]
        # steering: -1.0 to 1.0
        # throttle: 0.0 to 1.0
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # Storage for received data
        self.data = {
            'ips': None,
            'imu_orientation': None,
            'imu_angular_velocity': None,
            'lidar_ranges': None,
            'left_encoder': None,
            'right_encoder': None,
            'speed': None,
            'steering': None,
            'collision_count': None,
            'lap_time': None,
            'lap_count': None,
        }
        
        # Previous values for computing deltas
        self.prev_data = {
            'collision_count': 0,
            'lap_count': 0,
            'position': None,
            'lap_time': 0,
        }
        
        # Encoder tracking
        self.prev_encoder = {
            'left': None,
            'right': None,
            'timestamp': None
        }
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.create_subscription(Point, '/autodrive/roboracer_1/ips', 
                                self.ips_callback, qos_profile)
        self.create_subscription(Imu, '/autodrive/roboracer_1/imu', 
                                self.imu_callback, qos_profile)
        self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', 
                                self.lidar_callback, qos_profile)
        self.create_subscription(JointState, '/autodrive/roboracer_1/left_encoder', 
                                self.left_encoder_callback, qos_profile)
        self.create_subscription(JointState, '/autodrive/roboracer_1/right_encoder', 
                                self.right_encoder_callback, qos_profile)
        self.create_subscription(Float32, '/autodrive/roboracer_1/speed', 
                                self.speed_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/steering', 
                                self.steering_callback, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', 
                                self.collision_count_callback, 10)
        self.create_subscription(Float32, '/autodrive/roboracer_1/lap_time', 
                                self.lap_time_callback, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', 
                                self.lap_count_callback, 10)
        
        # Create publishers for commands
        self.reset_pub = self.create_publisher(Bool, '/autodrive/reset_command', 10)
        self.steering_cmd_pub = self.create_publisher(Float32, 
                                '/autodrive/roboracer_1/steering_command', 10)
        self.throttle_cmd_pub = self.create_publisher(Float32, 
                                '/autodrive/roboracer_1/throttle_command', 10)
        
        # Start ROS 2 spinning in a separate thread
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Wait for initial data
        self.get_logger().info('Waiting for initial sensor data...')
        time.sleep(2.0)
        self.get_logger().info('Environment initialized!')
    
    def _spin_ros(self):
        """Spin ROS 2 in a separate thread"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
    
    # ROS 2 Callbacks
    def ips_callback(self, msg):
        self.data['ips'] = [msg.x, msg.y, msg.z]
    
    def imu_callback(self, msg):
        self.data['imu_orientation'] = [msg.orientation.x, msg.orientation.y, 
                                       msg.orientation.z, msg.orientation.w]
        self.data['imu_angular_velocity'] = [msg.angular_velocity.x, 
                                            msg.angular_velocity.y, 
                                            msg.angular_velocity.z]
    
    def lidar_callback(self, msg):
        self.data['lidar_ranges'] = np.array(msg.ranges)
    
    def left_encoder_callback(self, msg):
        if len(msg.position) > 0:
            current_time = time.time()
            position = msg.position[0]
            
            velocity = None
            if len(msg.velocity) > 0 and msg.velocity[0] != 0.0:
                velocity = msg.velocity[0]
            elif self.prev_encoder['left'] is not None and self.prev_encoder['timestamp'] is not None:
                dt = current_time - self.prev_encoder['timestamp']
                if dt > 0:
                    velocity = (position - self.prev_encoder['left']) / dt
            
            self.data['left_encoder'] = velocity
            self.prev_encoder['left'] = position
            self.prev_encoder['timestamp'] = current_time
    
    def right_encoder_callback(self, msg):
        if len(msg.position) > 0:
            position = msg.position[0]
            
            velocity = None
            if len(msg.velocity) > 0 and msg.velocity[0] != 0.0:
                velocity = msg.velocity[0]
            elif self.prev_encoder['right'] is not None:
                current_time = time.time()
                if self.prev_encoder['timestamp'] is not None:
                    dt = current_time - self.prev_encoder['timestamp']
                    if dt > 0:
                        velocity = (position - self.prev_encoder['right']) / dt
            
            self.data['right_encoder'] = velocity
            self.prev_encoder['right'] = position
    
    def speed_callback(self, msg):
        self.data['speed'] = msg.data
    
    def steering_callback(self, msg):
        self.data['steering'] = msg.data
    
    def collision_count_callback(self, msg):
        self.data['collision_count'] = msg.data
    
    def lap_time_callback(self, msg):
        self.data['lap_time'] = msg.data
    
    def lap_count_callback(self, msg):
        self.data['lap_count'] = msg.data
    
    def _get_sparse_lidar(self, num_readings=36):
        """Get sparse LiDAR readings (36 rays from 1081 total)"""
        if self.data['lidar_ranges'] is None:
            return np.full(num_readings, 10.0, dtype=np.float32)
        
        ranges = self.data['lidar_ranges']
        total_rays = len(ranges)  # Should be 1081
        
        # Sample exactly num_readings evenly spaced indices
        indices = np.linspace(0, total_rays - 1, num_readings, dtype=int)
        sparse_ranges = ranges[indices]
        
        # Replace inf/nan with max range
        sparse_ranges = np.nan_to_num(sparse_ranges, nan=10.0, posinf=10.0, neginf=0.0)
        
        return sparse_ranges.astype(np.float32)
    
    def _get_observation(self):
        """Construct observation vector"""
        # LiDAR (36 readings)
        lidar = self._get_sparse_lidar(36)
        
        # Position and orientation
        if self.data['ips'] is not None:
            pose_x = self.data['ips'][0]
            pose_y = self.data['ips'][1]
        else:
            pose_x = 0.0
            pose_y = 0.0
        
        if self.data['imu_orientation'] is not None:
            _, _, pose_theta = quaternion_to_euler(*self.data['imu_orientation'])
        else:
            pose_theta = 0.0
        
        pose = np.array([pose_x, pose_y, pose_theta], dtype=np.float32)
        
        # Steering and speed
        steering = self.data['steering'] if self.data['steering'] is not None else 0.0
        speed = self.data['speed'] if self.data['speed'] is not None else 0.0
        
        steering_arr = np.array([steering], dtype=np.float32)
        speed_arr = np.array([speed], dtype=np.float32)
        
        # Angular velocity
        angular_vel_z = (self.data['imu_angular_velocity'][2] 
                        if self.data['imu_angular_velocity'] is not None else 0.0)
        angular_vel_arr = np.array([angular_vel_z], dtype=np.float32)
        
        # Encoder velocities
        left_enc_vel = self.data['left_encoder'] if self.data['left_encoder'] is not None else 0.0
        right_enc_vel = self.data['right_encoder'] if self.data['right_encoder'] is not None else 0.0
        encoders = np.array([left_enc_vel, right_enc_vel], dtype=np.float32)
        
        # Collision count (normalized)
        collision_count = (self.data['collision_count'] / 100.0 
                          if self.data['collision_count'] is not None else 0.0)
        collision_arr = np.array([collision_count], dtype=np.float32)
        
        # Lap time (normalized to 0-1, assuming max 300 seconds)
        lap_time = (self.data['lap_time'] / 300.0 
                   if self.data['lap_time'] is not None else 0.0)
        lap_time_arr = np.array([lap_time], dtype=np.float32)
        
        # Lap count
        lap_count = float(self.data['lap_count'] if self.data['lap_count'] is not None else 0)
        lap_count_arr = np.array([lap_count], dtype=np.float32)
        
        # Concatenate all observations
        obs = np.concatenate([
            lidar,  # 36
            pose,  # 3
            steering_arr,  # 1
            speed_arr,  # 1
            angular_vel_arr,  # 1
            encoders,  # 2
            collision_arr,  # 1
            lap_time_arr,  # 1
            lap_count_arr  # 1
        ]).astype(np.float32)
        
        # Verify shape
        assert obs.shape == (47,), f"Observation shape mismatch: {obs.shape}, expected (47,)"
        
        return obs
    
    def _compute_reward(self):
        """Compute reward based on current state"""
        reward = 0.0
        
        # Speed reward: encourage forward movement
        speed = self.data['speed'] if self.data['speed'] is not None else 0.0
        reward += speed * 0.1
        
        # Collision penalty
        if self.data['collision_count'] is not None:
            collision_delta = self.data['collision_count'] - self.prev_data['collision_count']
            if collision_delta > 0:
                reward -= 10.0 * collision_delta
        
        # Lap completion reward
        if self.data['lap_count'] is not None:
            # lap_delta = self.data['lap_count'] - self.prev_data['lap_count']
            lap_delta = self.data['lap_count']
            if lap_delta > 0:
                reward += 100.0 * lap_delta
        
        # Progress reward: penalize staying still
        if self.data['ips'] is not None and self.prev_data['position'] is not None:
            current_pos = np.array(self.data['ips'][:2])
            prev_pos = np.array(self.prev_data['position'][:2])
            distance = np.linalg.norm(current_pos - prev_pos)
            reward += distance * 1.0
        
        # Penalty for getting too close to obstacles
        lidar = self._get_sparse_lidar(36)
        min_distance = np.min(lidar)
        if min_distance < 0.5:
            reward -= (0.5 - min_distance) * 5.0
        
        # Small penalty for high steering (encourage smooth driving)
        if self.data['steering'] is not None:
            reward -= abs(self.data['steering']) * 0.01
        
        return reward
    
    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Episode ends if too many collisions
        if self.data['collision_count'] is not None and self.data['collision_count'] > 20:
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Send reset command (set to True)
        reset_msg = Bool()
        reset_msg.data = True
        self.reset_pub.publish(reset_msg)
        self.get_logger().info('Sent reset command: True')
        
        # Small delay to let the reset trigger
        time.sleep(0.1)
        
        # Set reset back to False to allow normal operation
        reset_msg.data = False
        self.reset_pub.publish(reset_msg)
        self.get_logger().info('Sent reset command: False')
        
        # Wait for reset to complete and system to stabilize
        time.sleep(1.5)
        
        # Reset internal state
        self.current_step = 0
        self.prev_data = {
            'collision_count': self.data['collision_count'] if self.data['collision_count'] is not None else 0,
            'lap_count': self.data['lap_count'] if self.data['lap_count'] is not None else 0,
            'position': self.data['ips'][:2] if self.data['ips'] is not None else None,
            'lap_time': self.data['lap_time'] if self.data['lap_time'] is not None else 0,
        }
        self.prev_encoder = {
            'left': None,
            'right': None,
            'timestamp': None
        }
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Unpack action
        steering = float(action[0])
        throttle = float(action[1])
        
        # Publish action
        steering_msg = Float32()
        steering_msg.data = steering
        self.steering_cmd_pub.publish(steering_msg)
        
        throttle_msg = Float32()
        throttle_msg.data = throttle
        self.throttle_cmd_pub.publish(throttle_msg)
        
        # Wait for simulator to process
        time.sleep(0.05)  # 20 Hz control rate
        
        # Update previous state
        if self.data['collision_count'] is not None:
            self.prev_data['collision_count'] = self.data['collision_count']
        if self.data['lap_count'] is not None:
            self.prev_data['lap_count'] = self.data['lap_count']
        if self.data['ips'] is not None:
            self.prev_data['position'] = self.data['ips'][:2]
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        self.current_step += 1
        done = self._is_done()
        truncated = False
        
        # Info dictionary
        info = {
            'collision_count': self.data['collision_count'],
            'lap_count': self.data['lap_count'],
            'lap_time': self.data['lap_time'],
            'speed': self.data['speed'],
        }
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render is handled by the simulator"""
        pass
    
    def close(self):
        """Clean up resources"""
        self.destroy_node()

class CustomWandbCallback(BaseCallback):
    """Custom callback for logging additional metrics to W&B"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_collisions = []
        self.episode_laps = []
        self.episode_lap_times = []
        
    def _on_step(self) -> bool:
        # Log every step metrics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                # Episode finished
                if 'episode' in info:
                    ep_reward = info['episode']['r']
                    ep_length = info['episode']['l']
                    
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Log episode metrics
                    wandb.log({
                        'episode/reward': ep_reward,
                        'episode/length': ep_length,
                        'episode/reward_mean_10': np.mean(self.episode_rewards[-10:]),
                        'episode/length_mean_10': np.mean(self.episode_lengths[-10:]),
                    }, step=self.num_timesteps)
                
                # Log custom info
                if 'collision_count' in info:
                    self.episode_collisions.append(info['collision_count'])
                    wandb.log({
                        'episode/collisions': info['collision_count'],
                        'episode/collisions_mean_10': np.mean(self.episode_collisions[-10:]) if self.episode_collisions else 0,
                    }, step=self.num_timesteps)
                
                if 'lap_count' in info:
                    self.episode_laps.append(info['lap_count'])
                    wandb.log({
                        'episode/laps': info['lap_count'],
                        'episode/laps_mean_10': np.mean(self.episode_laps[-10:]) if self.episode_laps else 0,
                    }, step=self.num_timesteps)
                
                if 'lap_time' in info:
                    if info['lap_time'] > 0:
                        self.episode_lap_times.append(info['lap_time'])
                        wandb.log({
                            'episode/lap_time': info['lap_time'],
                            'episode/lap_time_mean_10': np.mean(self.episode_lap_times[-10:]),
                        }, step=self.num_timesteps)
                
                if 'speed' in info and info['speed'] is not None:
                    wandb.log({
                        'episode/speed': info['speed'],
                    }, step=self.num_timesteps)
        
        return True

def train_ppo(total_timesteps=100000, 
              save_dir='./ppo_roboracer/',
              project_name='roboracer-ppo',
              run_name=None,
              use_wandb=True):
    """Train PPO agent with wandb logging"""
    
    # Initialize wandb
    if use_wandb:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                'total_timesteps': total_timesteps,
                'algorithm': 'PPO',
                'policy': 'MlpPolicy',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_steps_per_episode': 1000,
                'observation_space': 47,
                'action_space': 2,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    
    # Create environment
    env = RoboracerEnv(max_steps=1000)
    
    # Check environment
    print("Checking environment...")
    check_env(env)
    
    # Wrap environment
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_dir,
        name_prefix='ppo_roboracer',
        verbose=1
    )
    
    callbacks = [checkpoint_callback]
    
    if use_wandb:
        # Add both standard WandbCallback and custom callback
        wandb_callback = WandbCallback(
            model_save_path=f"{save_dir}/wandb_models",
            verbose=2,
        )
        custom_callback = CustomWandbCallback(verbose=1)
        callbacks.extend([wandb_callback, custom_callback])
    
    # Create PPO model with TensorBoard logging (W&B will sync it)
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{save_dir}/tensorboard/"  # Always enable for W&B sync
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    if use_wandb:
        print(f"Logging to W&B project: {project_name}")
        print(f"View your run at: {run.url}")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_dir}/ppo_roboracer_final")
    env.save(f"{save_dir}/vec_normalize.pkl")
    
    if use_wandb:
        # Log final statistics
        wandb.log({
            'training/final_timesteps': total_timesteps,
            'training/completed': True,
        })
        
        # Save model as wandb artifact
        model_artifact = wandb.Artifact(
            f"ppo_roboracer_{run.id}", 
            type="model",
            description="Final PPO model for Roboracer"
        )
        model_artifact.add_file(f"{save_dir}/ppo_roboracer_final.zip")
        model_artifact.add_file(f"{save_dir}/vec_normalize.pkl")
        run.log_artifact(model_artifact)
        
        wandb.finish()
    
    print(f"Training complete! Model saved to {save_dir}")
    
    return model, env


def evaluate_ppo(model_path='./ppo_roboracer/ppo_roboracer_final.zip',
                 vec_normalize_path='./ppo_roboracer/vec_normalize.pkl',
                 n_episodes=10,
                 render=True):
    """Evaluate trained PPO agent in the simulator"""
    
    print(f"Loading model from {model_path}")
    
    # Create environment
    env = RoboracerEnv(max_steps=1000)
    env = DummyVecEnv([lambda: env])
    
    # Load normalization statistics
    try:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Don't normalize rewards during evaluation
        print(f"Loaded normalization stats from {vec_normalize_path}")
    except FileNotFoundError:
        print("Warning: No normalization stats found, using unnormalized environment")
    
    # Load model
    model = PPO.load(model_path, env=env)
    print(f"Model loaded successfully!")
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_collisions = []
    episode_laps = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        print(f"\n[Episode {episode + 1}/{n_episodes}] Starting...")
        
        while not done:
            # Predict action (deterministic for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # print debug info about action
            print(f"Step {episode_length + 1}: Action: Steering={action[0][0]:.3f}, Throttle={action[0][1]:.3f}")

            # Take action
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if render:
                time.sleep(0.01)  # Slow down for visualization
        
        # Extract info from last step
        final_info = info[0] if isinstance(info, list) else info
        collisions = final_info.get('collision_count', 0)
        laps = final_info.get('lap_count', 0)
        lap_time = final_info.get('lap_time', 0)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_collisions.append(collisions)
        episode_laps.append(laps)
        
        print(f"[Episode {episode + 1}] Reward: {episode_reward:.2f}, "
              f"Length: {episode_length}, Collisions: {collisions}, "
              f"Laps: {laps}, Lap Time: {lap_time:.2f}s")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Mean Collisions: {np.mean(episode_collisions):.2f} ± {np.std(episode_collisions):.2f}")
    print(f"Mean Laps: {np.mean(episode_laps):.2f} ± {np.std(episode_laps):.2f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Worst Reward: {min(episode_rewards):.2f}")
    print("="*60)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_collisions': episode_collisions,
        'episode_laps': episode_laps,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
    }



if __name__ == '__main__':
    import sys
    import argparse

    eval = True
    parser = argparse.ArgumentParser(description='Train or evaluate PPO agent for Roboracer')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluation mode (default: training mode)')
    parser.add_argument('--timesteps', type=int, default=100000, 
                       help='Total training timesteps')
    parser.add_argument('--save-dir', type=str, default='./ppo_roboracer/',
                       help='Directory to save/load models')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for evaluation (default: save-dir/ppo_roboracer_final.zip)')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--project', type=str, default='roboracer-ppo',
                       help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='W&B run name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging (training only)')
    
    args = parser.parse_args()
    
    try:
        if eval:
            # Evaluation mode
            print("="*60)
            print("EVALUATION MODE")
            print("="*60)
            
            model_path = args.model_path or f"{args.save_dir}/ppo_roboracer_final.zip"
            vec_normalize_path = f"{args.save_dir}/vec_normalize.pkl"
            
            results = evaluate_ppo(
                model_path=model_path,
                vec_normalize_path=vec_normalize_path,
                n_episodes=args.n_eval_episodes,
                render=True
            )
        else:
            # Training mode
            print("="*60)
            print("TRAINING MODE")
            print("="*60)
            
            model, env = train_ppo(
                total_timesteps=args.timesteps,
                save_dir=args.save_dir,
                project_name=args.project,
                run_name=args.run_name,
                use_wandb=not args.no_wandb
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()