import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import copy

from model import DQN
from Replay_Buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        player,
        nrow=6,
        ncol=7,
        action_dim=7,
        lr=3e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=200_000,
        batch_size=64,
        target_update_freq=500,
        device="cpu"
    ):

        self.device = torch.device(device)

        # Set the player
        assert(player in [-1,1])
        self.player = player
        
        # Board info
        self.nrow = nrow
        self.ncol = ncol
        self.state_dim = nrow * ncol
        self.action_dim = action_dim

        # Online Networks
        self.online_net = DQN(nrow, ncol, action_dim).to(self.device)
        
        # Target Networks
        self.target_net = DQN(nrow, ncol, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Historical Opponent Network
        self.opponent_net = DQN(nrow,ncol,action_dim).to(self.device)
        self.opponent_net.load_state_dict(self.online_net.state_dict())
        self.opponent_net.eval()
        
        #Optimizer - Adam
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        
        #Reply Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.step_count = 0
        self.total_reward = 0
        
    # =========================================================
    # Load Historical Checkpoint Opponent
    # =========================================================
    def load_opponent_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.opponent_net.load_state_dict(
            checkpoint["online_model"]
        )

        self.opponent_net.eval()

    # =========================================================
    # Frozen Opponent Update
    # =========================================================
    def update_opponent(self):

        self.opponent_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.opponent_net.eval()

    # =========================================================
    # Self-play Action Selection
    # =========================================================
    def select_action_self_play(
        self,
        state,
        valid_actions
    ):

        state = np.array(state)

        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.opponent_net(state)

        q_values = q_values.squeeze().cpu().numpy()

        # Mask invalid actions
        masked_q = np.full_like(
            q_values,
            -np.inf
        )

        masked_q[valid_actions] = q_values[
            valid_actions
        ]

        return int(np.argmax(masked_q))

    def update_reward(self, reward):
        self.total_reward += reward

    def select_action(self, state, valid_actions):
    
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
    
        state = torch.tensor(
            state,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            q_values = self.online_net(state)
    
        q_values = q_values.squeeze().cpu().numpy()
    
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
    
        return int(np.argmax(masked_q))
            
    # Training Step
    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            valid_actions_batch

        ) = self.replay_buffer.sample(
            self.batch_size
        )
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Current Q
        q_values = self.online_net(states).gather(1, actions)

        
        # =========================
        # TARGET COMPUTATION
        # =========================
        with torch.no_grad():
            # -------------------------
            # 1. ACTION SELECTION (online net)
            # -------------------------
            next_q_online = self.online_net(next_states)
        
            masked_online_q = torch.full_like(next_q_online, -1e9)
            for i, valid_actions in enumerate(valid_actions_batch):
                masked_online_q[i, valid_actions] = next_q_online[i, valid_actions]
        
            next_actions = masked_online_q.argmax(1, keepdim=True)
        
            # -------------------------
            # 2. ACTION EVALUATION (target net)
            # -------------------------
            next_q_target = self.target_net(next_states)
            max_next_q = next_q_target.gather(1, next_actions)
        
            # -------------------------
            # 3. FINAL TARGET
            # -------------------------
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Loss
        loss = F.mse_loss(q_values, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            10.0
        )
        
        self.optimizer.step()
        
        # Step Counter
        self.step_count += 1

        # Target network update
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
