# Rubik's Cube Gymnasium Environment

This repository contains a custom Gymnasium (formerly OpenAI Gym) environment for the Rubik's Cube, designed for testing Reinforcement Learning (RL) agents and various cube-solving algorithms. The environment uses a sticker-based representation for the cube's state, providing a straightforward interface for RL experiments.

## Features

* **Sticker-Based State Representation:** The cube's state is represented as a 1D NumPy array of 54 integers, where each integer corresponds to the color of a specific sticker (0-5).
* **Discrete Action Space:** Supports 12 fundamental Rubik's Cube rotations (6 faces, clockwise and counter-clockwise).
* **Reward Function:** Provides a large positive reward upon solving the cube and a small negative reward for each step taken, encouraging efficient solutions.
* **Random Scrambling:** The environment can be initialized with a randomly scrambled cube, allowing for diverse starting conditions for learning and testing.
* **Text-Based Rendering:** Offers a simple `ansi` (ASCII art) rendering mode to visualize the cube's state in the console.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/SHIM-JINSEO/rubikCube.git](https://github.com/SHIM-JINSEO/rubikCube.git)
    cd rubikCube
    ```

2.  **Install Dependencies:**
    This project requires `gymnasium` and `numpy`.
    ```bash
    pip install gymnasium numpy
    ```

## Usage

Here's how you can use the Rubik's Cube environment in your Python code:

```python
import gymnasium as gym
import numpy as np
from rubiks_cube_env import RubiksCubeEnv # Assuming your environment class is in rubiks_cube_env.py

# 1. Create the environment
#    scramble_depth: Number of random moves to scramble the cube at reset.
#    render_mode: "human" for console output per step, "ansi" for returning a string, None for no rendering.
env = RubiksCubeEnv(scramble_depth=10, render_mode="human")

# 2. Reset the environment to get an initial scrambled state
observation, info = env.reset()
print("--- Initial (Scrambled) State ---")
env.render() # Render the initial state

# 3. Interact with the environment
total_reward = 0
terminated = False
truncated = False
step_count = 0

print("\n--- Performing Random Moves ---")
while not terminated and not truncated:
    action = env.action_space.sample() # Take a random action (for demonstration)
                                       # In an RL agent, this would be determined by the policy.

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    print(f"Step: {step_count}, Action: {action}, Reward: {reward:.1f}, Solved: {terminated}")
    env.render() # Render the cube state after each step

    if step_count > 1000: # Prevent infinite loops for unsolved cubes
        print("Max steps reached. Stopping simulation.")
        truncated = True

if terminated:
    print(f"\nCube Solved in {step_count} steps! Total Reward: {total_reward:.1f}")
elif truncated:
    print(f"\nEpisode truncated after {step_count} steps. Total Reward: {total_reward:.1f}")

# 4. Close the environment (important for proper cleanup)
env.close()

# Example: Verify specific moves (F -> F' should return to solved state)
print("\n--- Verifying F -> F' Sequence ---")
env_test = RubiksCubeEnv(scramble_depth=0, render_mode="human") # Start from solved state
obs_test, info_test = env_test.reset()
print("Solved State:")
env_test.render()

print("\nApplying F (Action 0):")
obs_test, _, _, _, _ = env_test.step(0)
env_test.render()

print("\nApplying F' (Action 1):")
obs_test, reward_final, terminated_final, _, _ = env_test.step(1)
env_test.render()

if terminated_final:
    print("Cube successfully returned to solved state after F -> F'!")
else:
    print("Cube did NOT return to solved state. There might be an issue in the action mappings.")
env_test.close()
