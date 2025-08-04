import magiccube
import numpy as np
from main import RubiksCubeEnv
from magiccube import BasicSolver

initial_state = "RYRWYGRRRGOGGRGGGGYBYYGYYYYBRBBOBBBBWRWWBWWRWOOOOWOOWO"
cube = magiccube.Cube(3, initial_state)
solver = BasicSolver(cube)
solver.solve()
history = np.array(cube.history())
extracted_action_sequence = history[:np.random.choice(len(history))]

cube.reset()
cube.set(initial_state)
cube.rotate(extracted_action_sequence)
start_str = cube.get()
solver.solve()
end_str = cube.get()
actions = cube.history()

env = RubiksCubeEnv(cube_size=3, start_state=start_str, end_state=end_str, autoscrambled=False, limit_max_steps=True)
print("\n*******************************************")
print("start cube state:")
env.get_start_state()
print(f"Is cube solved?: {env._is_solved()}")

print("end(goal) cube state:")
env.get_end_state()
print(f"Is cube solved?: If cube reach end state, should be True")
print("*******************************************")

for _ in range(env.max_steps):
    current_action = str(actions[env.current_step])
    state, terminated, truncated = env.apply_action(current_action)
    print(f"\nStep {env.current_step}: Action={actions[env.current_step]}")
    env.render() # check the cube state after action
    print(f"Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        print("\n*******************************************")
        print(f"End episode. total steps: {env.current_step}")
        print("final cube state:")
        env.render()
        print(f"Cube solved or not: {env._is_solved()}")

        print("\nGoal state:")
        env.get_end_state()
        print("*******************************************")
        break