import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RubiksCubeEnv(gym.Env): # sticker-based representation
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4} # render_modes 추가

    def __init__(self, scramble_depth=20, render_mode=None):
        super().__init__() # initalize gym.Env
        self.scramble_depth = scramble_depth # inital scramble depth
        self.render_mode = render_mode # set render mode (human or ansi)

        # cube color (ex): W, Y, R, O, B, G)
        # 0: White (W), 1: Yellow (Y), 2: Red (R), 3: Orange (O), 4: Blue (B), 5: Green (G)
        self.colors = ['W', 'Y', 'R', 'O', 'B', 'G']
        self.num_colors = len(self.colors) # 6 colors

        # define state space: each color of 54 stikers (0~5)
        # 6 face * 9 sticker = 54 stickers
        self.observation_space = spaces.Box(low=0, high=self.num_colors - 1,
                                            shape=(54,), dtype=np.uint8)

        # ation_space : 12 actions
        # 0: F (Front Clockwise)
        # 1: F' (Front Counter-Clockwise)
        # 2: B (Back Clockwise)
        # 3: B' (Back Counter-Clockwise)
        # 4: U (Up Clockwise)
        # 5: U' (Up Counter-Clockwise)
        # 6: D (Down Clockwise)
        # 7: D' (Down Counter-Clockwise)
        # 8: L (Left Clockwise)
        # 9: L' (Left Counter-Clockwise)
        # 10: R (Right Clockwise)
        # 11: R' (Right Counter-Clockwise)
        self.action_space = spaces.Discrete(12)

        # current state of cube (array of 54 sticker color)
        self.state = None

        # solved state of cube
        self.solved_state = self._get_solved_state()

        # location mapping for each action
        # (this is the core logic of the Rubik's Cube rotation)
        # this is a manual mapping of the Rubik's Cube rotation principles.
        # each sticker is represented by an index from 0 to 53,
        # index in of each face:
        # Front: 0-8
        # Up: 9-17
        # Right: 18-26
        # Left: 27-35
        # Back: 36-44
        # Down: 45-53

        # mapping sticker rotation(define as dictionary)
        # key: action index, value: index array for new location which each sticker move on
        self._action_mappings = self._create_action_mappings()

    def _get_solved_state(self):
        solved = np.zeros(54, dtype=np.uint8)
        for i in range(6):
            start_idx = i * 9
            end_idx = start_idx + 9
            solved[start_idx:end_idx] = i
        return solved

    def _create_action_mappings(self):
        mappings = {}

        # initial sticker state (0~53)
        initial_indices = np.arange(54)
        # index of each face
        F_face = np.array([0,1,2,3,4,5,6,7,8])
        U_face = np.array([9,10,11,12,13,14,15,16,17])
        R_face = np.array([18,19,20,21,22,23,24,25,26])
        L_face = np.array([27,28,29,30,31,32,33,34,35])
        B_face = np.array([36,37,38,39,40,41,42,43,44])
        D_face = np.array([45,46,47,48,49,50,51,52,53])

        # clockwise rotation mapping
        def rotate_face_clockwise(face_indices):
            """
            0 1 2   ->   6 3 0
            3 4 5   ->   7 4 1
            6 7 8   ->   8 5 2
            """
            new_indices = np.array([
                face_indices[6], face_indices[3], face_indices[0],
                face_indices[7], face_indices[4], face_indices[1],
                face_indices[8], face_indices[5], face_indices[2]
            ])
            return new_indices

        # counter-clockwise rotation mapping
        def rotate_face_counter_clockwise(face_indices):
            """
            0 1 2   ->   2 5 8
            3 4 5   ->   1 4 7
            6 7 8   ->   0 3 6
            """
            new_indices = np.array([
                face_indices[2], face_indices[5], face_indices[8],
                face_indices[1], face_indices[4], face_indices[7],
                face_indices[0], face_indices[3], face_indices[6]
            ])
            return new_indices
        
       # --- 0: F (Front Clockwise) ---
        f_map = np.copy(initial_indices)
        f_map[F_face] = rotate_face_clockwise(F_face)
        # around stiker moving: U(15,16,17) -> R(24,25,26) -> D(47,46,45) -> L(33,32,31) -> U(15,16,17)
        # (D, L face: inverse order)
        f_map[[15,16,17, 24,25,26, 47,46,45, 33,32,31]] = \
              [24,25,26, 47,46,45, 33,32,31, 15,16,17]
        mappings[0] = f_map

        # --- 1: F' (Front Counter-Clockwise) ---
        fp_map = np.copy(initial_indices)
        fp_map[F_face] = rotate_face_counter_clockwise(F_face)
        # around stiker moving (inverse order of F): U(15,16,17) -> L(33,32,31) -> D(47,46,45) -> R(24,25,26) -> U(15,16,17)
        fp_map[[15,16,17, 33,32,31, 47,46,45, 24,25,26]] = \
               [33,32,31, 47,46,45, 24,25,26, 15,16,17]
        mappings[1] = fp_map

        # --- 2: B (Back Clockwise) ---
        b_map = np.copy(initial_indices)
        b_map[B_face] = rotate_face_clockwise(B_face)
        # around stiker moving: U(9,10,11) -> L(27,28,29) -> D(53,52,51) -> R(20,19,18) -> U(9,10,11)
        b_map[[9,10,11, 27,28,29, 53,52,51, 20,19,18]] = \
              [27,28,29, 53,52,51, 20,19,18, 9,10,11]
        mappings[2] = b_map

        # --- 3: B' (Back Counter-Clockwise) ---
        bp_map = np.copy(initial_indices)
        bp_map[B_face] = rotate_face_counter_clockwise(B_face)
        # around stiker moving (inverse order of B): U(9,10,11) -> R(20,19,18) -> D(53,52,51) -> L(27,28,29) -> U(9,10,11)
        bp_map[[9,10,11, 20,19,18, 53,52,51, 27,28,29]] = \
               [20,19,18, 53,52,51, 27,28,29, 9,10,11]
        mappings[3] = bp_map

        # --- 4: U (Up Clockwise) ---
        u_map = np.copy(initial_indices)
        u_map[U_face] = rotate_face_clockwise(U_face)
        # around stiker moving: F(0,1,2) -> R(18,19,20) -> B(36,37,38) -> L(27,28,29) -> F(0,1,2)
        u_map[[0,1,2, 18,19,20, 36,37,38, 27,28,29]] = \
              [18,19,20, 36,37,38, 27,28,29, 0,1,2]
        mappings[4] = u_map

        # --- 5: U' (Up Counter-Clockwise) ---
        up_map = np.copy(initial_indices)
        up_map[U_face] = rotate_face_counter_clockwise(U_face)
        # around stiker moving (inverse order of U): F(0,1,2) -> L(27,28,29) -> B(36,37,38) -> R(18,19,20) -> F(0,1,2)
        up_map[[0,1,2, 27,28,29, 36,37,38, 18,19,20]] = \
               [27,28,29, 36,37,38, 18,19,20, 0,1,2]
        mappings[5] = up_map

        # --- 6: D (Down Clockwise) ---
        d_map = np.copy(initial_indices)
        d_map[D_face] = rotate_face_clockwise(D_face)
        # around sticker moving: F(6,7,8) -> L(35,34,33) -> B(44,43,42) -> R(26,25,24) -> F(6,7,8)
        # (L, B, R: inverse order)
        d_map[[6,7,8, 35,34,33, 44,43,42, 26,25,24]] = \
              [35,34,33, 44,43,42, 26,25,24, 6,7,8]
        mappings[6] = d_map

        # --- 7: D' (Down Counter-Clockwise) ---
        dp_map = np.copy(initial_indices)
        dp_map[D_face] = rotate_face_counter_clockwise(D_face)
        # around sticker moving (inverse order of D): F(6,7,8) -> R(26,25,24) -> B(44,43,42) -> L(35,34,33) -> F(6,7,8)
        dp_map[[6,7,8, 26,25,24, 44,43,42, 35,34,33]] = \
               [26,25,24, 44,43,42, 35,34,33, 6,7,8]
        mappings[7] = dp_map

        # --- 8: L (Left Clockwise) ---
        l_map = np.copy(initial_indices)
        l_map[L_face] = rotate_face_clockwise(L_face)
        # around sticker moving: U(9,12,15) -> F(0,3,6) -> D(45,48,51) -> B(44,41,38) -> U(9,12,15)
        # (B: inverse order)
        l_map[[9,12,15, 0,3,6, 45,48,51, 44,41,38]] = \
              [0,3,6, 45,48,51, 44,41,38, 9,12,15]
        mappings[8] = l_map

        # --- 9: L' (Left Counter-Clockwise) ---
        lp_map = np.copy(initial_indices)
        lp_map[L_face] = rotate_face_counter_clockwise(L_face)
        # around sticker moving (inverse order of L): U(9,12,15) -> B(44,41,38) -> D(45,48,51) -> F(0,3,6) -> U(9,12,15)
        lp_map[[9,12,15, 44,41,38, 45,48,51, 0,3,6]] = \
               [44,41,38, 45,48,51, 0,3,6, 9,12,15]
        mappings[9] = lp_map

        # --- 10: R (Right Clockwise) ---
        r_map = np.copy(initial_indices)
        r_map[R_face] = rotate_face_clockwise(R_face)
        # around sticker moving: U(17,14,11) -> B(36,39,42) -> D(53,50,47) -> F(8,5,2) -> U(17,14,11)
        # (U, B, D, F: inverse order)
        r_map[[17,14,11, 36,39,42, 53,50,47, 8,5,2]] = \
              [36,39,42, 53,50,47, 8,5,2, 17,14,11]
        mappings[10] = r_map

        # --- 11: R' (Right Counter-Clockwise) ---
        rp_map = np.copy(initial_indices)
        rp_map[R_face] = rotate_face_counter_clockwise(R_face)
        # around sticker (inverse order of R): U(17,14,11) -> F(8,5,2) -> D(53,50,47) -> B(36,39,42) -> U(17,14,11)
        rp_map[[17,14,11, 8,5,2, 53,50,47, 36,39,42]] = \
               [8,5,2, 53,50,47, 36,39,42, 17,14,11]
        mappings[11] = rp_map

        return mappings

    def _apply_action(self, state, action_idx):
        """
        retrun new state after applying the action to the current state.
        """
        if action_idx not in self._action_mappings:
            raise ValueError(f"Invalid action index: {action_idx}")

        new_state = np.zeros_like(state)
        for old_idx, new_idx in enumerate(self._action_mappings[action_idx]):
            new_state[new_idx] = state[old_idx]
        return new_state

    def _is_solved(self, state):
        """check cube is solved."""
        for i in range(6):
            face_start_idx = i * 9
            face_colors = state[face_start_idx : face_start_idx + 9]
            if not np.all(face_colors == face_colors[0]):
                return False
        return True

    def reset(self, seed=None, options=None):
        """intialize environment and return initial state."""
        super().reset(seed=seed)

        self.state = np.copy(self.solved_state)
        self.current_step = 0

        # initial scramble
        for _ in range(self.scramble_depth):
            action = self.np_random.choice(self.action_space.n) # select random action
            self.state = self._apply_action(self.state, action)

        observation = self.state.copy()
        info = {} # additional info (if needed)

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        do agent action and return next state, reward, terminated, truncated, info.
        """
        self.current_step += 1

        # calculate next state app;lying the action
        self.state = self._apply_action(self.state, action)
        observation = self.state.copy()

        # calculate reward and termination
        reward = -0.1 # small negative reward for each step(time penalty)
        terminated = False # cube solved or not
        truncated = False # time out

        if self._is_solved(self.state):
            reward = 100.0 # huge reward for solving the cube
            terminated = True

        info = {} # aditional info (if needed)

        # if you need to render cube for each step use below
        """
        if self.render_mode == "human":
            self._render_frame()
        """
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """
        visualize the current state of the Rubik's Cube.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_ansi(self):
        """text-based rendering of the Rubik's Cube state."""
        s = self.state

        output = []

        # Up (U): 인덱스 9-17
        output.append("      " + " ".join([self.colors[s[9]], self.colors[s[10]], self.colors[s[11]]]))
        output.append("      " + " ".join([self.colors[s[12]], self.colors[s[13]], self.colors[s[14]]]))
        output.append("      " + " ".join([self.colors[s[15]], self.colors[s[16]], self.colors[s[17]]]))

        # Left (L), Front (F), Right (R), Back (B)
        # L: 27-35, F: 0-8, R: 18-26, B: 36-44
        for i in range(3):
            line = []
            line.extend([self.colors[s[27+i*3]], self.colors[s[28+i*3]], self.colors[s[29+i*3]]]) # Left
            line.extend([self.colors[s[0+i*3]], self.colors[s[1+i*3]], self.colors[s[2+i*3]]])   # Front
            line.extend([self.colors[s[18+i*3]], self.colors[s[19+i*3]], self.colors[s[20+i*3]]]) # Right
            line.extend([self.colors[s[36+i*3]], self.colors[s[37+i*3]], self.colors[s[38+i*3]]]) # Back (이 부분은 보통 큐브 평면도에서 잘림)
            output.append(" ".join(line))
        
        # Down (D): index 45-53
        output.append("      " + " ".join([self.colors[s[45]], self.colors[s[46]], self.colors[s[47]]]))
        output.append("      " + " ".join([self.colors[s[48]], self.colors[s[49]], self.colors[s[50]]]))
        output.append("      " + " ".join([self.colors[s[51]], self.colors[s[52]], self.colors[s[53]]]))
        
        return "\n".join(output)

    def _render_frame(self):
        """
        render the current state of the Rubik's Cube in human-readable format.
        GUI will be implemented later.
        """
        if self.render_mode == "human":
            print("\n--- Current Cube State ---")
            print(self._render_ansi())
            print("--------------------------\n")

    def close(self):
        pass 


# exanple usage
if __name__ == "__main__":
    # create environment (random scrample 20회, human render mode)
    env = RubiksCubeEnv(scramble_depth=20, render_mode="human")

    # reset environment (initial scramlbe state)
    observation, info = env.reset()
    print("Initial (Scrambled) State:")
    # print(observation) # print number array
    env.render()

    # agent do random action until cube is solved
    total_reward = 0
    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        action = env.action_space.sample() # select random action
        # print(f"\nStep {step_count + 1}: Applying action {action} ({env.action_space.n} available actions)")
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print('\n')
        print(f"Step {step_count}, Action: {action}, Reward: {reward:.1f}, Terminated: {terminated}")
        env.render() # render for each state
        print('\n')


        if terminated:
            print(f"\nCube Solved in {step_count} steps! Total Reward: {total_reward:.1f}")
            break
        if step_count > 100: # prevent infinite loop
            print(f"\nMax steps reached ({step_count}). Not solved. Total Reward: {total_reward:.1f}")
            break

    env.close()

    # check solved state
    print("\n--- Solved State Check ---")
    solved_env = RubiksCubeEnv(scramble_depth=0, render_mode="human")
    obs, info = solved_env.reset()
    solved_env.close()