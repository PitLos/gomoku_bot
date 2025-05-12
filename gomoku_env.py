import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GomokuEnv(gym.Env):
    """
    Gomoku (Five in a Row) environment for two-player zero-sum game.
    Extended: square-win (2x2) has higher reward than line win.
    Board is represented as a 2D numpy array with values:
     0 = empty, 1 = current player's stones, -1 = opponent's stones.
    Players alternate turns until one gets a win or the board is full.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, board_size=15, render_mode=None):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode

        # Observation space: board_size x board_size with values in {-1, 0, 1}
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.board_size, self.board_size),
            dtype=np.int8
        )

        # Action space: one of board_size*board_size positions
        self.action_space = spaces.Discrete(self.board_size * self.board_size)

        self.board = None
        self.current_player = 1  # 1 or -1
        self._winner = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self._winner = None
        return self.board.copy(), {}

    def step(self, action):
        row = action // self.board_size
        col = action % self.board_size

        # Illegal move: immediate loss
        if self.board[row, col] != 0:
            return self.board.copy(), -1, True, False, {"illegal_move": True}

        # Place stone
        self.board[row, col] = self.current_player

        # Check for 2x2 square win (higher reward)
        if self._check_square(row, col, self.current_player):
            self._winner = self.current_player
            return self.board.copy(), 2, True, False, {"winner": self.current_player, "square_win": True}

        # Check for five in a row
        if self._check_five_in_row(row, col, self.current_player):
            self._winner = self.current_player
            return self.board.copy(), 1, True, False, {"winner": self.current_player}

        # Check for draw
        if not (self.board == 0).any():
            return self.board.copy(), 0, True, False, {"draw": True}

        # Switch player
        self.current_player *= -1
        return self.board.copy(), 0, False, False, {}

    def render(self):
        line = '+' + ('-' * (2 * self.board_size - 1)) + '+'
        print(line)
        for r in range(self.board_size):
            row_str = '|'
            for c in range(self.board_size):
                val = self.board[r, c]
                if val == 1:
                    row_str += 'X '
                elif val == -1:
                    row_str += 'O '
                else:
                    row_str += '. '
            print(row_str.strip() + '|')
        print(line)

    def _check_five_in_row(self, row, col, player):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # forward
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # backward
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def _check_square(self, row, col, player):
        # Check all 2x2 blocks containing (row, col)
        for dr in (0, -1):
            for dc in (0, -1):
                r0, c0 = row + dr, col + dc
                if 0 <= r0 < self.board_size - 1 and 0 <= c0 < self.board_size - 1:
                    block = self.board[r0:r0+2, c0:c0+2]
                    if np.all(block == player):
                        return True
        return False

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
