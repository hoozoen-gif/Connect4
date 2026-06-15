import logging
import numpy as np
from gymnasium import spaces, Env
from Connect4Game import Connect4
import random

class Connect4Env(Env):
    def __init__(self, opponent_type="random"):
        
        # Board Size
        self.nrow = 6
        self.ncol = 7
        self.opponent_type = opponent_type
        # Action Space: choose the column
        self.action_space = spaces.Discrete(self.ncol)
            
        # Observation Space
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(2, self.nrow, self.ncol),
            dtype=np.float32
        )
        
        self.agent = None
        
        # Game State
        self.board = None       
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Connect4(self.nrow, self.ncol)
        self.current_player = 1
        return self._get_obs(), {}
 
        
    def _get_obs(self):
    
        board = np.array(
            self.game.board,
            dtype=np.float32
        )
    
        board = board * self.current_player
    
        own_pieces = (board == 1).astype(np.float32)
        opp_pieces = (board == -1).astype(np.float32)
    
        return np.stack(
            [own_pieces, opp_pieces],
            axis=0
        )

    def count_n_in_row(self, board, player, n): #Counting connections
        count = 0
        # horizontal
        for r in range(self.nrow):
            for c in range(self.ncol - (n-1)):
                if sum(board[r][c+i] == player for i in range(n)) == n:
                    count += 1
        # vertical
        for r in range(self.nrow - (n-1)):
            for c in range(self.ncol):
                if sum(board[r+i][c] == player for i in range(n)) == n:
                    count += 1
        # diag /
        for r in range(self.nrow - (n-1)):
            for c in range(self.ncol - (n-1)):
                if sum(board[r+i][c+i] == player for i in range(n)) == n:
                    count += 1
        # diag \
        for r in range(n-1, self.nrow):
            for c in range(self.ncol - (n-1)):
                if sum(board[r-i][c+i] == player for i in range(n)) == n:
                    count += 1

        return count
        
    def can_win_next_move(self, board, player):

        for col in range(self.ncol):

            # Skip full column
            if board[0][col] != 0:
                continue

            row = None

            # Find empty row
            for r in range(self.nrow - 1, -1, -1):

                if board[r][col] == 0:
                    row = r
                    break

            if row is None:
                continue

            # Simulate move
            temp_board = board.copy()

            temp_board[row][col] = player

            temp_game = Connect4(self.nrow, self.ncol)

            temp_game.board = temp_board

            # Check win
            if temp_game.win_check(player):
                return True

        return False
        
    def _shape_reward(self, prev_board, new_board, player):
        opponent = -player
        reward = 0.0
    
        def diff(p, n, weight):

            return (
                self.count_n_in_row(new_board, p, n)
                - self.count_n_in_row(prev_board, p, n)
            ) * weight

        # +0.02 Reward if the player move connects 2
        reward += diff(player, 2, 0.7)
        # 0.1 Bigger Reward if the player move connects 3
        reward += diff(player, 3, 1.5)
    
        # DEFENSIVE PENALTIES
        reward -= diff(opponent, 2, 1.0)
        reward -= diff(opponent, 3, 2.5)

        opponent_threat_before = self.can_win_next_move(
            prev_board,
            opponent
        )

        opponent_threat_after = self.can_win_next_move(
            new_board,
            opponent
        )
        
        # SUCCESSFULLY BLOCKED OPPONENT
        if opponent_threat_before and not opponent_threat_after:
            reward += 2.0
            
        # FAILED TO BLOCK THREAT
        if opponent_threat_after:
            reward -= 1.5

        return reward
        
    def get_winning_moves(self, board, player):

        winning_moves = []
    
        for col in range(self.ncol):
    
            # Full column
            if board[0][col] != 0:
                continue
    
            row = None
    
            for r in range(self.nrow - 1, -1, -1):
    
                if board[r][col] == 0:
                    row = r
                    break
    
            if row is None:
                continue
    
            temp_board = board.copy()
    
            temp_board[row][col] = player
    
            temp_game = Connect4(self.nrow, self.ncol)
    
            temp_game.board = temp_board
    
            if temp_game.win_check(player):
                winning_moves.append(col)
    
        return winning_moves
        
    def heuristic_opponent_action(self, opponent):
        player = -opponent
        valid_moves = [c for c in range(self.ncol) if self.game.is_valid_move(c)]
    
        # =========================
        # 1. WIN IF POSSIBLE
        # =========================
        for col in valid_moves:
            temp = Connect4(self.nrow, self.ncol)
            temp.board = self.game.board.copy()
            temp.current_player = opponent
            temp.move(col)
            if temp.win_check(opponent):
                return col
    
        # =========================
        # 2. BLOCK PLAYER WIN
        # =========================
        for col in valid_moves:
            temp = Connect4(self.nrow, self.ncol)
            temp.board = self.game.board.copy()
            temp.current_player = player
            temp.move(col)
            if temp.win_check(player):
                return col
    
        # CENTER PRIORITY
        center = self.ncol // 2
        if center in valid_moves:
            return center

        # RANDOM FALLBACK
        return random.choice(valid_moves)
        
    def step(self, action):
    
        terminated = False
        truncated = False
        
        player = 1
        opponent = -1
    
        # REWARD CONSTANTS
        WIN_REWARD = 10.0
        LOSS_REWARD = -10.0
        DRAW_REWARD = 0.5
        INVALID_PENALTY = -5.0
        STEP_PENALTY = -0.01
    
        reward = 0.0
    
        # =========================
        # 1. INVALID MOVE
        # =========================
        if not self.game.is_valid_move(action):
            return self._get_obs(), INVALID_PENALTY, True, truncated, {}
    
        # Save board BEFORE move
        prev_board = self.game.board.copy()
    
        # Check if opponent had winning threat BEFORE move
        opponent_threat = self.can_win_next_move(
            self.game.board,
            opponent
        )
        
        opponent_winning_moves = self.get_winning_moves(
            self.game.board,
            opponent
        )
        # =========================
        # 2. AGENT MOVE
        # =========================
        self.game.current_player = player
        self.game.move(action)
    
        # WIN
        if self.game.win_check(player):
            return (
                self._get_obs(),
                WIN_REWARD,
                True,
                truncated,
                {"result": "win"}
            )
        # if self.game.win_check(player):
        #     return self._get_obs(), WIN_REWARD, True, truncated, {}
   
        # DRAW
        if self.game.is_draw():
            return (
                self._get_obs(),
                DRAW_REWARD,
                True,
                truncated,
                {"result": "draw"}
            )

        if opponent_winning_moves:
            # Agent blocked correctly
            if action in opponent_winning_moves:
                reward += 3.0
        
            # Agent ignored forced block
            else:
                reward -= 5.0
     
        # =========================
        # SHAPING (AFTER AGENT MOVE)
        # =========================
        reward += self._shape_reward(prev_board, self.game.board, player)
            
        # =========================
        # OPPONENT MOVE
        # =========================
        valid_moves = [c for c in range(self.ncol) if self.game.is_valid_move(c)]
    
        self.game.current_player = opponent
    
        # =========================
        # SELF-PLAY SUPPORT
        # =========================
        if self.opponent_type == "heuristic":

            opponent_action = self.heuristic_opponent_action(opponent)

        elif self.opponent_type == "self_play":

            opponent_action = self.agent.select_action_self_play(
                self._get_obs(),
                valid_moves
            )

        else:
            opponent_action = random.choice(valid_moves)

        self.game.move(opponent_action)
    
        # OPPONENT WIN
        if self.game.win_check(opponent):
            return (
                self._get_obs(),
                LOSS_REWARD,
                True,
                truncated,
                {"result": "loss"}
            )
        # if self.game.win_check(opponent):
        #     return self._get_obs(), LOSS_REWARD, True, truncated, {}
    
        # DRAW
        if self.game.is_draw():
            return (
                self._get_obs(),
                DRAW_REWARD,
                True,
                truncated,
                {"result": "draw"}
            )
    
        # =========================
        # 6. STEP PENALTY
        # =========================
        reward += STEP_PENALTY
    
        # Restore perspective
        self.game.current_player = player
        self.current_player = player
    
        return self._get_obs(), reward, terminated, truncated, {}
