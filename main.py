import gymnasium as gym
from gymnasium import spaces
import numpy as np
import magiccube

class RubiksCubeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, cube_size=3, max_steps=100, scramble_depth=10):
        super().__init__()
        self.cube_size = cube_size
        self.cube = magiccube.Cube(self.cube_size)
        self.max_steps = max_steps
        self.scramble_depth = scramble_depth # 큐브를 섞을 깊이

        # define action space: rotation of magiccube
        # ex: R, U, F, L, D, B, R', U', F', L', D', B', R2, U2, F2, L2, D2, B2
        self.actions = [
            "R", "U", "F", "L", "D", "B",
            "R'", "U'", "F'", "L'", "D'", "B'",
            "R2", "U2", "F2", "L2", "D2", "B2"
        ]
        self.action_space = spaces.Discrete(len(self.actions))

    
        # Y=0, R=1, G=2, O=3, B=4, W=5
        num_stickers = 6 * (self.cube_size ** 2)
        self.observation_space = spaces.Box(low=0, high=5, shape=(num_stickers,), dtype=np.int32) 

        self.current_step = 0

    def _get_obs(self):
        color_map = {'Y': 0, 'R': 1, 'G': 2, 'O': 3, 'B': 4, 'W': 5}
        flat_state_str = self.cube.get()
        obs = np.array([color_map[char] for char in flat_state_str])
        return obs

    def _is_solved(self):
        return self.cube.is_done()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.cube = magiccube.Cube(self.cube_size) # reset the cube to a solved state
        
        # scramble the cube randomly
        scramble_moves = self.cube.scramble(self.scramble_depth)
        # print(f"Scrambled with: {scramble_moves}") # print scrabble moves for debugging

        self.current_step = 0
        observation = self._get_obs()
        info = {} #optional information
        return observation, info

    def step(self, action):
        action_str = self.actions[action]
        self.cube.rotate(action_str)

        self.current_step += 1

        observation = self._get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {} # optional information

        if self._is_solved():
            reward = 1000 # huge reward for solving the cube
            terminated = True
        elif self.current_step >= self.max_steps:
            reward = -100 # out of time penalty
            truncated = True
        else:
            reward = -1 # small penalty for each step taken

        return observation, reward, terminated, truncated, info

    def render(self):
        print("\n" + str(self.cube))

    def close(self):
        pass

if __name__ == "__main__":
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