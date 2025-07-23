# Rubik's Cube Gymnasium Environment

This repository contains a custom Gymnasium (formerly OpenAI Gym) environment for the Rubik's Cube, designed for testing AI agents and various cube-solving algorithms. The environment uses Magiccube library for the cube's state, providing a straightforward interface.

## Features

* **State Representation with MagicCube library:** The cube's state is represented as a 1D NumPy array of integers, where each integer corresponds to the color of a specific sticker (0-5).
* **Discrete Action Space:** Supports all the possible Rubik's Cube rotations (including clockwise and counter-clockwise).
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
    This project requires `gymnasium` and `maggiccube`.
    ```bash
    pip install gymnasium magiccube
    ```

## Usage

Here's how you can use the Rubik's Cube environment in your Python code:

```python
    # /test.py
    from main import RubiksCubeEnv
    env = RubiksCubeEnv(cube_size=3, max_steps=500, scramble_depth=15) 

    observation, info = env.reset()
    print("initial cube state:")
    env.render()
    print(f"initial observation: {observation}")
    print(f"Is cube solved?: {env._is_solved()}")

    for _ in range(env.max_steps):
        total_reward = 0
        # select random action (it will be replaced with a real agent later)
        action = env.action_space.sample() 
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {env.current_step}: Action={env.actions[action]}")
        env.render() # check the cube state after action
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            print(f"\nEnd episode. total steps: {env.current_step}, total reward: {total_reward}")
            print("final cube state:")
            env.render()
            print(f"Cube solved or not: {env._is_solved()}")
            break
    
    env.close()
```

## Moves of cube
For NxN rubik's cube, moves can be classified into three types
### Outer Layer Moves(Basic moves of 6 face)
- F, F', F2 (Front face)
- B, B', B2 (Back face)
- U, U', U2 (Up face)
- D, D', D2 (Down face)
- L, L', L2 (Left face)
- R, R', R2 (Right face)
### Wide moves
It means **rotate  external_layer~x at the same time**.
It is depend on size(called N) of cube. Suppose x is loacation of layer starting from external layer(current layer).
If N is even, 1 < x <= N/2 and if N is odd, 1 < x <= (N+1)/2
- x Fw, x Fw', x Fw2
- x Bw, x Bw', x Bw2 
- x Uw, x Uw', x Uw2 
- x Dw, x Dw', x Dw2 
- x Lw, x Lw', x Lw2 
- x Rw, x Rw', x Rw2
### Single Slice Moves
It means rotate a one internal layer. 1 < x < N
- x F, x F', x F2 
- x B, x B', x B2 
- x U, x U', x U2 
- x D, x D', x D2 
- x L, x L', x L2 
- x R, x R', x R2
