from main import RubiksCubeEnv
import numpy as np

env = RubiksCubeEnv(cube_size=3, max_steps=500) 

state = env.scramble()
print("initial cube state:")
env.render()
print(f"Is cube solved?: {env._is_solved()}")

for _ in range(env.max_steps):
    # select random action (it will be replaced with a real agent later)
    action = np.random.choice(env.actions)
    state, terminated, truncated = env.apply_action(action)
    
    print(f"\nStep {env.current_step}: Action={action}")
    env.render() # check the cube state after action
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print(f"\nEnd episode. total steps: {env.current_step}")
        print("final cube state:")
        env.render()
        print(f"Cube solved or not: {env._is_solved()}")
        break

env.close()