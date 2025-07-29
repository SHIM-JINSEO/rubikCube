import numpy as np
import magiccube

class RubiksCubeEnv():

    def __init__(self, cube_size=3, start_state=None, end_state=None, autoscrambled=True, limit_max_steps=False,):
        #If auto scramble is true, the initial and goal state are automatically determined
        self.cube_size = cube_size
        self.max_steps = 500
        self.limit_max_steps = limit_max_steps
        self.cube = magiccube.Cube(self.cube_size)
        self.start_cube = magiccube.Cube(self.cube_size) 
        self.end_cube = magiccube.Cube(self.cube_size)
        
        if autoscrambled:# if autoscrambled is True, the start and end states are randomly generated
            self.cube.scramble()
            self.state = self.get_state()
            self.start_state = self.state
            self.start_cube.set(self.start_state)
            self.end_cube.scramble()
            self.end_state = self.end_cube.get()
        elif start_state is not None and end_state is not None: # If start_state and end_state are provided, set them directly
            self.cube.set(start_state)
            self.start_cube.set(start_state)
            self.end_cube.set(end_state)
            self.state = self.get_state()
            self.start_state = self.get_start_state()
            self.end_state = self.get_end_state()

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
        return self.cube.get()
    
    def _is_solved(self):
        return self.state == self.end_state

    def reset(self):
        self.cube = magiccube.Cube(self.cube_size)
        self.start_cube = magiccube.Cube(self.cube_size) 
        self.end_cube = magiccube.Cube(self.cube_size)
        
        self.cube.scramble()
        self.state = self.get_state()
        self.start_state = self.state
        self.start_cube.set(self.start_state)

        self.end_cube.scramble()
        self.end_state = self.end_cube.get()

        self.actions = self.possible_actions()
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
        elif self.limit_max_steps and self.current_step >= self.max_steps:
            truncated = True

        return state, terminated, truncated

    def render(self):
        print("\n" + str(self.cube))

    def get_start_state(self):
        print("\n" + str(self.start_cube))
        return self.start_cube.get()
    
    def get_end_state(self):
        print("\n" + str(self.end_cube))
        return self.end_cube.get()
    
if __name__ == "__main__":
    start_str = (
    'RRRRRRRRR' +  # F (Front) - R 
    'WWWWWWWWW' +  # U (Up) - W 
    'BBBBBBBBB' +  # R (Right) - B 
    'GGGGGGGGG' +  # L (Left) - G 
    'OOOOOOOOO' +  # B (Back) - O 
    'YYYYYYYYY'    # D (Down) - Y 
    )   
    end_str = (
    'RRRRRRWWW' +  # F (Front)
    'WWWWWWGGG' +  # U (Up) 
    'BBBBBBBYY' +  # R (Right) 
    'GGGGGGGOO' +  # L (Left) 
    'OOOOOOOOO' +  # B (Back) 
    'BBBBYYYYY'    # D (Down) 
    )
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
        # select random action (it will be replaced with a real agent later)
        action = np.random.choice(env.actions)
        state, terminated, truncated = env.apply_action(action)
        
        print(f"\nStep {env.current_step}: Action={action}")
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
    