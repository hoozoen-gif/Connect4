import logging
import numpy as np
from gymnasium import spaces, Env
from Connect4Game import Connect4
import random

class Connect4Env(Env):
    def __init__(self):
        
        # Board Size
        self.nrow = 6
        self.ncol = 7
        
        # Action Space: choose the column
        self.action_space = spaces.Discrete(self.ncol)
            
        # Observation Space
        # 0 = empty
        # 1 = player 1
        # -1 = player 2
        self.observation_space = spaces.Box(
            low = -1,
            high = 1,
            shape = (self.nrow, self.ncol),
            dtype=np.int8
        )
            
        # Game State
        self.board = None       
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        self.game = Connect4(self.nrow, self.ncol)
        
        obs = np.array(self.game.board, dtype=np.int8)
        self.current_player = 1
        
        return obs, {}
        
    def _get_obs(self):
        # THE perspective of the CURRENT PLAYER
        return np.array(self.game.board, dtype=np.int8) * self.current_player
        
    def step(self, action):
        
        terminated = False
        truncated = False
        # Every step will give a minor punishment, so it will tend to win in smaller steps
        reward = 0.0
        player = self.current_player


        #Invalid move check
        #When there's a illegal move detected, it will apply a punishment, and will terminate the episode.
        if not self.game.is_valid_move(action):
            
            reward = -10.0
            terminated = True

            return self._get_obs(), reward, terminated, truncated, {}
            
        #Apply move, putting action into the move func in GAME
        self.game.move(action)
    
        # Win Check (Agent Wins)
        if self.game.win_check(player):
            reward = +20.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}


        # Draw Check
        if self.game.is_draw():
            reward = -0.5
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        self.game.switch_player()
        self.current_player = self.game.current_player

        # Check if opponent just won (i.e. previous player lost)
        if self.game.win_check(-self.current_player):
            reward = -20.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}
        
        #Small penalty
        reward = -0.01
            
        return self._get_obs(), reward, terminated, truncated, {}