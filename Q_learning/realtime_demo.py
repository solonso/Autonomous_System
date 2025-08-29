#!/usr/bin/env python3
"""
Improved Real-time Q-Learning Demo with TurtleBot Visualization
=============================================================

This script addresses the issues with:
1. Static images instead of real-time animation
2. TurtleBot image not being displayed properly
3. FIXED: Sparse rewards causing local minimum traps
4. FIXED: Aggressive epsilon decay limiting exploration
5. FIXED: Poor Q-value initialization
6. FIXED: Insufficient learning episodes and parameters

Key Improvements:
- Distance-based reward shaping (+0.1 closer, -0.05 away, +10.0 goal)
- Slower epsilon decay (0.999^episode vs 0.995^episode)
- Smart Q-initialization (goal state = 10.0)
- Better hyperparameters (α=0.15, γ=0.99, 1000 episodes)
- Extended max steps (150 vs 100)
- Performance tracking and success metrics

Features:
- Smooth real-time animation using matplotlib
- Proper TurtleBot image loading and display
- Interactive visualization during training
- Step-by-step policy demonstration
- Reward shaping for efficient learning
- Performance monitoring and statistics
- NEW: Comprehensive training analytics dashboard
- NEW: Policy visualization with action arrows
- NEW: Value function heatmaps
- NEW: Learning performance analysis
- NEW: Success rate tracking and learning phases analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import time
import os
import pickle
import argparse

class InlineScreenRecorder:
  """Minimal wrapper around Matplotlib's FFMpegWriter to record frames inline."""
  def __init__(self, figure, output_path, fps=30, dpi=100):
    self.figure = figure
    self.output_path = os.path.expanduser(output_path)
    self.fps = fps
    self.dpi = dpi
    self._context = None
    self._writer = None

  def start(self):
    os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    self._writer = FFMpegWriter(fps=self.fps)
    # Enter writer context so we can call grab_frame() at each render
    self._context = self._writer.saving(self.figure, self.output_path, dpi=self.dpi)
    self._context.__enter__()
    print(f"[Recorder] Started: {self.output_path} @ {self.fps} fps")

  def capture(self):
    if self._writer is not None:
      self._writer.grab_frame()

  def stop(self):
    if self._context is not None:
      try:
        self._context.__exit__(None, None, None)
      finally:
        self._context = None
        self._writer = None
        print(f"[Recorder] Finalized: {self.output_path}")

def save_qtable_to_file(agent, filename):
    """Save Q-table and training information to file"""
    save_data = {
        'q_table': agent.Q,
        'env_shape': agent.env.map.shape,
        'goal_position': agent.env.goal,
        'alpha': agent.alpha,
        'gamma': agent.gamma,
        'epsilon': agent.epsilon,
        'episode_rewards': getattr(agent, 'episode_rewards', []),
        'episode_steps': getattr(agent, 'episode_steps', [])
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Q-table and training data saved to {filename}")

def load_qtable_from_file(filename):
    """Load Q-table and training information from file"""
    try:
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
        print(f"Q-table loaded from {filename}")
        return save_data
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        return None

class RealtimeMapEnv:
  def __init__(self, map, goal, max_steps, robot_image_path='turtlebot.png'):
    self.map = map
    self.current_state = None
    self.goal = goal.astype(np.int32)
    self.actions = 4
    self.steps = 0
    self.max_steps = max_steps
    self.valid_rows, self.valid_cols = np.where(self.map == 0)
   
    # Load robot image with better error handling
    self.robot_image = self.load_robot_image(robot_image_path)
   
    # Initialize figure for real-time updates
    plt.ion() # Interactive mode
    self.fig, self.ax = plt.subplots(figsize=(10, 8))
    self.setup_environment()
    # Keep handles to dynamic artists to avoid full redraw-induced jitter
    self.robot_artist = None
    # Fixed-position text to avoid layout changes from title updates
    self.title_text = self.ax.text(0.02, 1.02,
                                   '', transform=self.ax.transAxes,
                                   fontsize=14, fontweight='bold', va='bottom')
    # Optional: inline recorder injected by caller
    self.recorder = None
   
    if map[goal[0], goal[1]] != 0:
      raise ValueError("Goal position is an obstacle")

  def load_robot_image(self, robot_image_path):
    """Load robot image with detailed feedback"""
    print(f"Attempting to load robot image: {robot_image_path}")
   
    if not os.path.exists(robot_image_path):
      print(f"ERROR: File not found: {robot_image_path}")
      print(f"Current directory: {os.getcwd()}")
      print(f"Files in current directory: {os.listdir('.')}")
      return None
     
    try:
      image = mpimg.imread(robot_image_path)
      print(f"Successfully loaded TurtleBot image!")
      print(f"  Image shape: {image.shape}")
      print(f"  Image dtype: {image.dtype}")
      print(f"  Value range: [{image.min():.3f}, {image.max():.3f}]")
     
      # Handle different image formats
      if len(image.shape) == 3 and image.shape[2] == 4:
        print("  RGBA image detected")
      elif len(image.shape) == 3 and image.shape[2] == 3:
        print("  RGB image detected")
      else:
        print(f"  Unusual image format: {image.shape}")
       
      return image
     
    except Exception as e:
      print(f"ERROR: Error loading image: {e}")
      return None

  def setup_environment(self):
    """Set up the visualization environment"""
    self.ax.clear()
   
    # Draw map
    self.ax.imshow(self.map, cmap='gray_r', origin='upper', alpha=0.8,
           extent=[-0.5, self.map.shape[1]-0.5, self.map.shape[0]-0.5, -0.5])
   
    # Draw goal
    goal_rect = Rectangle((self.goal[1]-0.4, self.goal[0]-0.4), 0.8, 0.8,
               linewidth=3, edgecolor='gold', facecolor='yellow', alpha=0.9)
    self.ax.add_patch(goal_rect)
   
    # Add grid
    for i in range(self.map.shape[0] + 1):
      self.ax.axhline(y=i-0.5, color='lightgray', linewidth=0.5, alpha=0.5)
    for j in range(self.map.shape[1] + 1):
      self.ax.axvline(x=j-0.5, color='lightgray', linewidth=0.5, alpha=0.5)
   
    # Set limits and labels
    self.ax.set_xlim(-0.5, self.map.shape[1]-0.5)
    self.ax.set_ylim(self.map.shape[0]-0.5, -0.5)
    self.ax.set_xlabel('X Position')
    self.ax.set_ylabel('Y Position')
    self.ax.set_xticks(range(self.map.shape[1]))
    self.ax.set_yticks(range(self.map.shape[0]))

  def reset(self):
    self.steps = 0
    random_idx = np.random.choice(len(self.valid_rows))
    self.current_state = np.array([self.valid_rows[random_idx], self.valid_cols[random_idx]])
    return self.current_state

  def step(self, action):
    # Calculate old distance for reward shaping
    old_distance = abs(self.current_state[0] - self.goal[0]) + abs(self.current_state[1] - self.goal[1])
    new_state = self.current_state.copy()
   
    if action == 0:  # up
      new_state[0] -= 1
    elif action == 1: # down
      new_state[0] += 1
    elif action == 2: # left
      new_state[1] -= 1
    elif action == 3: # right
      new_state[1] += 1

    # Check boundaries and obstacles
    if (new_state[0] < 0 or new_state[0] >= self.map.shape[0] or
      new_state[1] < 0 or new_state[1] >= self.map.shape[1] or
      self.map[new_state[0], new_state[1]] == 1):
      # Invalid move - stay in place with small penalty
      new_state = self.current_state.copy()
      reward = -0.1
    else:
      self.current_state = new_state.copy()
      # Calculate new distance for reward shaping
      new_distance = abs(new_state[0] - self.goal[0]) + abs(new_state[1] - self.goal[1])
     
      # Reward shaping based on distance change
      if new_distance < old_distance:
        reward = 0.1 # Moving closer to goal
      elif new_distance > old_distance:
        reward = -0.05 # Moving away from goal
      else:
        reward = -0.02 # Same distance (neutral move)

    self.steps += 1

    if np.array_equal(new_state, self.goal):
      reward = 10.0 # Large reward for reaching goal
      done = True
    elif self.steps >= self.max_steps:
      done = True
      reward += -1.0 # Additional penalty for timeout
    else:
      done = False

    return new_state, reward, done

  def render_realtime(self, episode=0, step=0, reward=0, delay=0.1):
    """Render with real-time animation (no full redraw to prevent jitter)."""
    # Draw or update robot artist
    if self.robot_image is not None:
      robot_size = 0.8
      extent = [self.current_state[1] - robot_size/2, self.current_state[1] + robot_size/2,
                self.current_state[0] + robot_size/2, self.current_state[0] - robot_size/2]
      if self.robot_artist is None or getattr(self.robot_artist, 'get_array', None) is None:
        self.robot_artist = self.ax.imshow(self.robot_image, extent=extent, zorder=10, alpha=0.9)
      else:
        self.robot_artist.set_extent(extent)
      print(f" TurtleBot at position ({self.current_state[0]}, {self.current_state[1]})")
    else:
      # Fallback: update scatter position
      if self.robot_artist is None:
        self.robot_artist = self.ax.scatter(self.current_state[1], self.current_state[0],
                                            c='red', s=400, marker='o', zorder=10,
                                            edgecolor='darkred', linewidth=3)
      else:
        # PathCollection set_offsets expects [[x, y]] in data coords
        self.robot_artist.set_offsets([[self.current_state[1], self.current_state[0]]])
      print(f" Robot (fallback) at position ({self.current_state[0]}, {self.current_state[1]})")

    # Update overlay text without relayout
    self.title_text.set_text(f' Real-time Q-Learning with TurtleBot\nEpisode: {episode}, Step: {step}, Reward: {reward:.1f}')

    # Update canvas
    self.fig.canvas.draw_idle()
    # Capture the drawn frame if recorder is active
    if getattr(self, 'recorder', None) is not None:
      self.recorder.capture()
    plt.pause(delay)

    return self.fig


class RealtimeQLearning:
  def __init__(self, env, alpha, gamma, epsilon, n_episodes):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_episodes = n_episodes
    # Initialize Q-table with small random values
    self.Q = np.random.rand(env.map.shape[0], env.map.shape[1], env.actions) * 0.01
    # Initialize Q-values for goal state to be high (attracts robot)
    self.Q[env.goal[0], env.goal[1], :] = 10.0

  def epsilon_greedy_policy(self, s, epsilon):
    if np.random.rand() < epsilon:
      return np.random.randint(0, self.env.actions)
    else:
      return np.argmax(self.Q[s[0], s[1]])

  def train_episode(self, episode_num, visualize=False, delay=0.1):
    s = self.env.reset()
    done = False
    total_reward = 0
    step_count = 0
   
    # Slower epsilon decay - maintains exploration longer
    epsilon = max(0.05, self.epsilon * (0.999 ** episode_num))
   
    if visualize:
      self.env.render_realtime(episode=episode_num, step=step_count, reward=total_reward, delay=delay)
   
    while not done:
      a = self.epsilon_greedy_policy(s, epsilon)
      s_prime, reward, done = self.env.step(a)
     
      # Q-learning update
      self.Q[s[0], s[1], a] += self.alpha * (reward + self.gamma * np.max(self.Q[s_prime[0], s_prime[1]]) - self.Q[s[0], s[1], a])
     
      s = s_prime
      total_reward += reward
      step_count += 1
     
      if visualize:
        self.env.render_realtime(episode=episode_num, step=step_count, reward=total_reward, delay=delay)
       
        if done and reward > 0:
          print(f" GOAL REACHED! Episode {episode_num}, Steps: {step_count}")
          time.sleep(1) # Celebrate!
   
    return total_reward, step_count

  def train(self, episodes_to_visualize=5, delay=0.2):
    print(" Starting Improved Real-time Q-Learning Training!")
    print(f" Parameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
    print(f" Will visualize first {episodes_to_visualize} episodes")
    print("=" * 60)
   
    episode_rewards = []
    episode_steps = []
   
    # Store as instance attributes for later access
    self.episode_rewards = episode_rewards
    self.episode_steps = episode_steps
   
    for episode in range(self.n_episodes):
      # Visualize only first few episodes
      visualize = episode < episodes_to_visualize
     
      reward, steps = self.train_episode(episode, visualize=visualize, delay=delay)
      episode_rewards.append(reward)
      episode_steps.append(steps)
     
      if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_steps = np.mean(episode_steps[-100:])
        epsilon = max(0.05, self.epsilon * (0.999 ** episode))
        print(f"Episode {episode+1:4d}/{self.n_episodes} | Avg Reward: {avg_reward:6.2f} | Avg Steps: {avg_steps:4.1f} | ε: {epsilon:.3f}")
       
        # Check if we're making progress
        if episode > 200 and avg_reward > 5:
          print(f" Good progress detected! Average reward: {avg_reward:.2f}")
   
    print(" Training completed!")
   
    # Print final statistics
    final_avg_reward = np.mean(episode_rewards[-50:])
    final_avg_steps = np.mean(episode_steps[-50:])
    successful_episodes = sum(1 for r in episode_rewards[-100:] if r > 5)
   
    print(f"\n FINAL RESULTS:")
    print(f"  Last 50 episodes avg reward: {final_avg_reward:.2f}")
    print(f"  Last 50 episodes avg steps: {final_avg_steps:.1f}")
    print(f"  Successful episodes (last 100): {successful_episodes}/100")
   
    # Plot comprehensive training analytics
    self.plot_training_progress(episode_rewards, episode_steps)

  def demonstrate_policy(self, delay=0.5):
    """Demonstrate the learned policy"""
    print("\n Demonstrating Learned Policy...")
   
    s = self.env.reset()
    done = False
    step = 0
    total_reward = 0
   
    print(f" Demo starting at {s}, goal at {self.env.goal}")
    self.env.render_realtime(episode="DEMO", step=step, reward=total_reward, delay=delay)
   
    while not done and step < 100: # Increased step limit
      # Use learned policy (no exploration)
      a = np.argmax(self.Q[s[0], s[1]])
      s, reward, done = self.env.step(a)
      total_reward += reward
      step += 1
     
      action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
      distance = abs(s[0] - self.env.goal[0]) + abs(s[1] - self.env.goal[1])
      print(f"Step {step:2d}: {action_names[a]:5s} → ({s[0]:2d},{s[1]:2d}), Distance: {distance:2.0f}, Reward: {reward:5.2f}")
     
      self.env.render_realtime(episode="DEMO", step=step, reward=total_reward, delay=delay)
   
    if total_reward > 5: # Updated success criterion
      print(f" SUCCESS! Reached goal in {step} steps with total reward {total_reward:.2f}!")
      
      # Keep recording for 5 seconds after reaching goal
      for _ in range(50):  # 50 frames * 0.1 seconds = 5 seconds
        if getattr(self.env, 'recorder', None) is not None:
          self.env.recorder.capture()
        plt.pause(0.1)
    else:
      print(f"ERROR: Failed to reach goal in {step} steps (total reward: {total_reward:.2f})")

  def plot_training_progress(self, episode_rewards, episode_steps):
    """Create comprehensive training analytics"""
    print("\n Generating training analytics plots...")
   
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    episodes = list(range(len(episode_rewards)))
   
    # Plot 1: Episode rewards
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.6, color='blue', linewidth=1)
   
    # Moving average for smoother trend
    window_size = 50
    if len(episode_rewards) >= window_size:
      moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
      axes[0, 0].plot(episodes[window_size-1:], moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
      axes[0, 0].legend()
   
    axes[0, 0].set_title(' Training Rewards Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
   
    # Plot 2: Path lengths (steps to goal)
    axes[0, 1].plot(episodes, episode_steps, alpha=0.6, color='green', linewidth=1)
   
    # Moving average for steps
    if len(episode_steps) >= window_size:
      moving_avg_steps = np.convolve(episode_steps, np.ones(window_size)/window_size, mode='valid')
      axes[0, 1].plot(episodes[window_size-1:], moving_avg_steps, color='darkgreen', linewidth=2, label=f'Moving Average ({window_size})')
      axes[0, 1].legend()
   
    axes[0, 1].set_title(' Steps to Goal Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps to Goal')
    axes[0, 1].grid(True, alpha=0.3)
   
    # Plot 3: Epsilon decay
    epsilons = [max(0.05, self.epsilon * (0.999 ** i)) for i in episodes]
    axes[1, 0].plot(episodes, epsilons, color='orange', linewidth=2)
    axes[1, 0].set_title(' Epsilon Decay (Exploration Rate)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True, alpha=0.3)
   
    # Plot 4: Value function heatmap
    value_function = self.get_value_function()
    im = axes[1, 1].imshow(value_function, cmap='viridis', origin='upper', aspect='auto')
    axes[1, 1].set_title(' Learned Value Function', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
   
    # Mark goal position
    axes[1, 1].scatter(self.env.goal[1], self.env.goal[0], c='gold', s=100, marker='*',
             edgecolor='black', linewidth=2, zorder=10)
   
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
   
    plt.suptitle(' Q-Learning Training Analytics Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

  def get_value_function(self):
    """Extract the value function (max Q-value for each state)"""
    value_function = np.max(self.Q, axis=2)
    # Mark obstacles as very low value
    value_function[self.env.map == 1] = np.min(value_function) - 1
    return value_function

  def get_optimal_policy(self):
    """Extract the optimal policy (best action for each state)"""
    policy = np.argmax(self.Q, axis=2)
    policy[self.env.map == 1] = -1 # Mark obstacles
    return policy

  def visualize_policy_arrows(self):
    """Visualize the learned policy as arrows on the map"""
    print("\n Generating policy visualization with arrows...")
   
    policy = self.get_optimal_policy()
    value_function = self.get_value_function()
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
   
    # Plot 1: Policy arrows
    ax1.imshow(self.env.map, cmap='gray_r', origin='upper', alpha=0.8)
   
    # Arrow directions for actions: up, down, left, right
    arrow_dirs = [(-0.3, 0), (0.3, 0), (0, -0.3), (0, 0.3)]
    colors = ['red', 'blue', 'green', 'orange']
    action_names = ['↑ Up', '↓ Down', '← Left', '→ Right']
   
    for i in range(self.env.map.shape[0]):
      for j in range(self.env.map.shape[1]):
        if self.env.map[i, j] == 0: # Free cell
          action = policy[i, j]
          if 0 <= action < 4: # Valid action
            dy, dx = arrow_dirs[action]
            ax1.arrow(j, i, dx, dy, head_width=0.15, head_length=0.15,
                fc=colors[action], ec=colors[action], alpha=0.8, linewidth=2)
   
    # Mark goal
    ax1.scatter(self.env.goal[1], self.env.goal[0], c='gold', s=400, marker='*',
          edgecolor='black', linewidth=3, zorder=10, label='Goal')
   
    # Create legend for actions
    for i, (color, name) in enumerate(zip(colors, action_names)):
      ax1.scatter([], [], c=color, s=100, marker='>', alpha=0.8, label=name)
   
    ax1.set_title(' Learned Policy (Action Arrows)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
   
    # Plot 2: Value function heatmap
    im = ax2.imshow(value_function, cmap='viridis', origin='upper', aspect='auto')
    ax2.scatter(self.env.goal[1], self.env.goal[0], c='gold', s=400, marker='*',
          edgecolor='black', linewidth=3, zorder=10)
    ax2.set_title(' State Value Function', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax2, shrink=0.8)
   
    plt.suptitle(' Q-Learning Policy and Value Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

  def analyze_learning_performance(self, episode_rewards, episode_steps):
    """Generate detailed performance analysis"""
    print("\n DETAILED LEARNING PERFORMANCE ANALYSIS")
    print("=" * 60)
   
    # Success rate analysis
    successful_episodes = [i for i, r in enumerate(episode_rewards) if r > 5]
    success_rate = len(successful_episodes) / len(episode_rewards) * 100
   
    print(f" Overall Success Rate: {success_rate:.1f}% ({len(successful_episodes)}/{len(episode_rewards)} episodes)")
   
    if successful_episodes:
      first_success = successful_episodes[0]
      print(f" First Success: Episode {first_success}")
     
      # Learning phases analysis
      early_phase = episode_rewards[:len(episode_rewards)//3]
      middle_phase = episode_rewards[len(episode_rewards)//3:2*len(episode_rewards)//3]
      late_phase = episode_rewards[2*len(episode_rewards)//3:]
     
      print(f"\n Learning Phases:")
      print(f"  Early (Episodes 1-{len(early_phase)}): Avg Reward = {np.mean(early_phase):.2f}")
      print(f"  Middle (Episodes {len(early_phase)+1}-{len(early_phase)+len(middle_phase)}): Avg Reward = {np.mean(middle_phase):.2f}")
      print(f"  Late (Episodes {len(early_phase)+len(middle_phase)+1}-{len(episode_rewards)}): Avg Reward = {np.mean(late_phase):.2f}")
     
      # Efficiency analysis
      if len([r for r in late_phase if r > 5]) > 0:
        successful_steps = [episode_steps[i] for i in range(len(episode_steps)) if episode_rewards[i] > 5]
        print(f"  Average steps when successful: {np.mean(successful_steps):.1f}")
        print(f"  Best performance: {min(successful_steps)} steps")
   
    print("\n Analysis complete!")
    return success_rate


def main():
  """Main demo function"""
  parser = argparse.ArgumentParser(description='Improved Real-time Q-Learning Demo')
  parser.add_argument('--record', action='store_true', help='Enable inline recording to MP4 using FFMpegWriter')
  parser.add_argument('--record-path', type=str, default=os.path.expanduser('~/Videos/q_learning_inline_%Y%m%d_%H%M%S.mp4'), help='Output path for the recording (supports strftime tokens)')
  parser.add_argument('--record-fps', type=int, default=30, help='Recording frames per second')
  parser.add_argument('--episodes', type=int, default=2000, help='Total training episodes')
  parser.add_argument('--viz-episodes', type=int, default=2, help='Number of early episodes to visualize')
  parser.add_argument('--delay', type=float, default=0.2, help='Render delay in seconds for visualization')
  args = parser.parse_args()

  # Define map
  map_data = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ]
 
  grid_map = np.array(map_data)
  goal_position = np.array([3, 17])
 
  print(" Improved Real-time Q-Learning with TurtleBot Demo")
  print(" Fixed: Sparse rewards, epsilon decay, Q-init, and parameters")
  print("=" * 60)
 
  # Create environment with more steps allowed
  env = RealtimeMapEnv(grid_map, goal_position, max_steps=150, robot_image_path='turtlebot.png')
 
  # Create agent with improved parameters
  agent = RealtimeQLearning(env, alpha=0.15, gamma=0.99, epsilon=0.9, n_episodes=args.episodes)
 
  # Optional inline recording setup
  recorder = None
  if args.record:
    # Expand strftime tokens in output path
    try:
      output_path = time.strftime(args.record_path)
    except Exception:
      output_path = args.record_path
    recorder = InlineScreenRecorder(figure=env.fig, output_path=output_path, fps=args.record_fps)
    env.recorder = recorder
    recorder.start()

  # Train with visualization
  agent.train(episodes_to_visualize=args.viz_episodes, delay=args.delay)
  
  # Save trained Q-table for later use in other environments
  save_qtable_to_file(agent, 'trained_qtable.pkl')
 
  # Generate comprehensive visualizations
  print("\n Generating comprehensive analysis visualizations...")
  agent.visualize_policy_arrows()
 
  # Analyze learning performance
  agent.analyze_learning_performance(agent.episode_rewards, agent.episode_steps)
 
  # Demonstrate final policy
  print("\n Final policy demonstration...")
  agent.demonstrate_policy(delay=0.5)

  # Stop recorder if active
  if recorder is not None:
    recorder.stop()
 
  # Keep plot open
  plt.ioff()
  plt.show()
 
  return env, agent


if __name__ == "__main__":
  env, agent = main() 