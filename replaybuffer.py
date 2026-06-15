import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        valid_actions
    ):
        #state      = board before move
        #action     = column chosen
        #reward     = +1 / -1 / 0
        #next_state = board after move
        #done       = game finished?

    
        # =========================
        # Original transition
        # =========================
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
                valid_actions
            )
        )
    
        # =========================
        # Mirrored transition
        # =========================
    
        mirrored_state = np.flip(
            state,
            axis=-1
        ).copy()
    
        mirrored_next_state = np.flip(
            next_state,
            axis=-1
        ).copy()
    
        mirrored_action = 6 - action
    
        mirrored_valid_actions = [
            6 - a
            for a in valid_actions
        ]
    
        mirrored_valid_actions.sort()
    
        self.buffer.append(
            (
                mirrored_state,
                mirrored_action,
                reward,
                mirrored_next_state,
                done,
                mirrored_valid_actions
            )
        )
        

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, valid_actions = zip(*batch)
    
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            valid_actions   # keep as list of lists (important)
        )

    def __len__(self):
        return len(self.buffer)
