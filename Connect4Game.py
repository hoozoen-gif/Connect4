import numpy as np

class Connect4:

    def __init__(self, n_rows=7, n_cols=6):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        self.current_player = 1

    def is_valid_move(self, col):
        # Checks in bounds/ column not full (By checking whether the TOP column occupied)
        return 0 <= col < self.n_cols and self.board[0][col] == 0

    def get_next_open_row(self, col):
        for row in range(self.n_rows - 1, -1, -1):
            if self.board[row][col] == 0:
                return row
        return None

    def move(self, col):
        if not self.is_valid_move(col):
            return False

        row = self.get_next_open_row(col)
        self.board[row][col] = self.current_player
        return True

    def switch_player(self):
        self.current_player = -self.current_player

    def win_check(self, player):
        board = self.board
        rows = self.n_rows
        cols = self.n_cols
        
        # Horizontal lines
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r][c+i] == player for i in range(4)):
                    return True

        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if all(board[r+i][c] == player for i in range(4)):
                    return True

        # Positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r+i][c+i] == player for i in range(4)):
                    return True

        # Negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r-i][c+i] == player for i in range(4)):
                    return True

        return False

    def is_draw(self):
        return np.all(self.board != 0)