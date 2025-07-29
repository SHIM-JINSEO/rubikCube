import numpy as np
import magiccube

class RubiksCubeEnv():

    def __init__(self, cube_size=3, max_steps=500):
        self.cube_size = cube_size
        self.max_steps = max_steps
        self.cube = magiccube.Cube(self.cube_size)
        self.state = self.get_state() 
       

        self.actions = self.possible_actions()

        self.face_size = self.cube_size ** 2
        self.num_stickers = 6 * self.face_size
        print(f"Number of stickers: {self.num_stickers}")

        self.current_step = 0

    def possible_actions(self):
        n = self.cube_size

        # define basic moves for outer layer
        outer_layer_moves = [
            "R", "U", "F", "L", "D", "B",
            "R'", "U'", "F'", "L'", "D'", "B'",
        ]
        
        # define wide moves
        '''
        # basic_wide_moves = [
        #     "Rw", "Uw", "Fw", "Lw", "Dw", "Bw",
        #     "Rw'", "Uw'", "Fw'", "Lw'", "Dw'", "Bw'",
        #     "Rw2", "Uw2", "Fw2", "Lw2", "Dw2", "Bw2"
        # ]
        
        # total_wide_moves = []
        # for i in range(2, (n+1)//2 + 1):
        #     # even: 1 < x <= n/2, odd: 1 < x <= (n+1)/2
        #     wide_moves = [str(i) + move for move in basic_wide_moves]
        #     total_wide_moves.extend(wide_moves)
        '''

        # define single slice moves
        total_slice_moves = []
        for i in range(2, n):
            # single slice moves: 1 < x < n
            slice_moves = [str(i) + move for move in outer_layer_moves] # e.g., "2R", "3U", etc.
            total_slice_moves.extend(slice_moves)
    
        possible_actions = np.array(outer_layer_moves + total_slice_moves)

        print(f"Possible actions: {possible_actions}")

        return possible_actions
    
    def size(self):
        return self.cube_size, self. face_size, self.num_stickers
    
    def get_state(self):
        color_map = {'Y': 0, 'R': 1, 'G': 2, 'O': 3, 'B': 4, 'W': 5}
        flat_state_str = self.cube.get()
        obs = np.array([color_map[char] for char in flat_state_str])
        return obs

    def _is_solved(self):
        return self.cube.is_done()

    def reset(self):
        self.cube = magiccube.Cube(self.cube_size) # reset the cube to a solved state
        self.current_step = 0
        state = self.get_state()
        return state

    def apply_action(self, action):
        if (not action in self.actions):
            print(f"It is Invalid action: {action}. Available actions: {self.actions}")
            return
        self.cube.rotate(action)
        self.current_step += 1

        state = self.get_state()
        terminated = False
        truncated = False

        if self._is_solved():
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        return state, terminated, truncated

    def render(self):
        print("\n" + str(self.cube))

    def close(self):
        pass
    
    def scramble(self, scramble_steps=20):
        self.cube.scramble(scramble_steps)
        return self.get_state()

if __name__ == "__main__":
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