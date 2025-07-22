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