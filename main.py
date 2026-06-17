import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import os
from Connect4Env import Connect4Env
from DQN_Agent import DQNAgent
from Connect4Game import Connect4
import pandas as pd

# =========================
# Generic Evaluation
# =========================
def evaluate_against_opponent(
    agent,
    env,
    opponent_type,
    n_games=50
):

    wins = 0
    losses = 0
    draws = 0

    # Save original settings
    original_epsilon = agent.epsilon
    original_opponent = env.opponent_type

    try:

        # No exploration during evaluation
        agent.epsilon = 0.0

        env.opponent_type = opponent_type

        for _ in range(n_games):

            state, _ = env.reset()

            done = False

            while not done:

                valid_actions = [
                    c
                    for c in range(env.ncol)
                    if env.game.is_valid_move(c)
                ]

                action = agent.select_action(
                    state,
                    valid_actions
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                done = terminated or truncated
                state = next_state

                if done:

                    result = info["result"]

                    if result == "win":
                        wins += 1

                    elif result == "loss":
                        losses += 1

                    else:
                        draws += 1

    finally:

        # Restore original settings
        agent.epsilon = original_epsilon
        env.opponent_type = original_opponent

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws / n_games
    }

def evaluate_against_checkpoint(
    agent,
    env,
    checkpoint_path,
    n_games=50
):

    wins = 0
    losses = 0
    draws = 0

    original_epsilon = agent.epsilon
    original_opponent_type = env.opponent_type

    original_opponent_weights = copy.deepcopy(
        agent.opponent_net.state_dict()
    )

    try:

        # Pure greedy evaluation
        agent.epsilon = 0.0

        checkpoint = torch.load(
            checkpoint_path,
            map_location=agent.device
        )

        agent.opponent_net.load_state_dict(
            checkpoint["online_model"]
        )

        env.opponent_type = "self_play"

        with torch.no_grad():

            for _ in range(n_games):

                state, _ = env.reset()

                done = False

                while not done:

                    valid_actions = [
                        c
                        for c in range(env.ncol)
                        if env.game.is_valid_move(c)
                    ]

                    action = agent.select_action(
                        state,
                        valid_actions
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info
                    ) = env.step(action)

                    done = terminated or truncated
                    state = next_state
                    if done:
                    
                        result = info.get("result", None)
                    
                        if result == "win":
                            wins += 1
                    
                        elif result == "loss":
                            losses += 1
                    
                        elif result == "draw":
                            draws += 1
                    
                        else:
                            print(
                                f"WARNING: Missing result. "
                                f"reward={reward}, info={info}"
                            )

    finally:

        # Restore original state
        agent.opponent_net.load_state_dict(
            original_opponent_weights
        )

        env.opponent_type = original_opponent_type

        agent.epsilon = original_epsilon

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "draw_rate": draws / n_games
    }

env = Connect4Env()
agent = DQNAgent(1)
agent.replay_buffer.buffer.clear()

## CONFIG ##

batch_size = 64
num_episodes = 15000

EVAL_INTERVAL = 100
EVAL_GAMES = 50

SAVE_INTERVAL = 500
MAX_CHECKPOINT_POOL = 5

# =========================
# Logging
# =========================
reward_history = []

heuristic_win_rates = []
random_win_rates = []
checkpoint_win_rates = []
checkpoint_eval_win_rates = []

loss_rates = []
draw_rates = []

heuristic_eval_episodes = []

training_logs = []
reward_window = []
# =========================
# Checkpoint pool
# =========================
checkpoint_pool = []

save_dir = "Saved_models"
os.makedirs(save_dir, exist_ok=True)

env.agent = agent

if __name__ == '__main__:
    # Code strts here

# =========================================================
# TRAINING LOOP
# =========================================================
for episode in range(1, num_episodes + 1):

    state, _ = env.reset()
    done = False
    total_reward = 0

    # =====================================================
    # OPPONENT SELECTION CURRICULUM
    # =====================================================

    r = random.random()

    # =====================================================
    # STAGE 1 : Episodes 1-2000
    # 70% Random
    # 30% Heuristic
    # =====================================================
    if episode <= 2000:

        if r < 0.70:
            env.opponent_type = "random"
        else:
            env.opponent_type = "heuristic"

    # =====================================================
    # STAGE 2 : Episodes 2001-4000
    # 40% Random
    # 40% Heuristic
    # 20% Self-Play
    # =====================================================
    elif episode <= 4000:

        if r < 0.40:
            env.opponent_type = "random"

        elif r < 0.80:
            env.opponent_type = "heuristic"

        else:
            env.opponent_type = "self_play"

            if len(checkpoint_pool) > 0:

                recent_pool = checkpoint_pool[-10:]

                selected_checkpoint = random.choice(
                    recent_pool
                )

                agent.load_opponent_checkpoint(
                    selected_checkpoint
                )

            else:
                agent.update_opponent()

    # =====================================================
    # STAGE 3 : Episodes 4001+
    # 10% Random
    # 20% Heuristic
    # 70% Self-Play
    # =====================================================
    else:

        if r < 0.10:
            env.opponent_type = "random"

        elif r < 0.30:
            env.opponent_type = "heuristic"

        else:
            env.opponent_type = "self_play"

            if len(checkpoint_pool) > 0:

                recent_pool = checkpoint_pool[-10:]

                selected_checkpoint = random.choice(
                    recent_pool
                )

                agent.load_opponent_checkpoint(
                    selected_checkpoint
                )

            else:
                agent.update_opponent()
            
    # =========================
    # Training game
    # =========================
    while not done:

        valid_actions = [
            c for c in range(env.ncol)
            if env.game.is_valid_move(c)
        ]

        action = agent.select_action(
            state,
            valid_actions
        )

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_valid_actions = [
            c for c in range(env.ncol)
            if env.game.is_valid_move(c)
        ]
        # Store transition
        agent.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
            next_valid_actions
        )

        state = next_state

        total_reward += reward
        
        # Train step
        agent.train()
        
    # Reward Logging
    reward_history.append(total_reward)
    reward_window.append(total_reward)
    
    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================
    if episode % SAVE_INTERVAL == 0:

        save_path = os.path.join(
            save_dir,
            f"model_ep_{episode}.pth"
        )

        torch.save({

            "online_model":
                agent.online_net.state_dict(),

            "target_model":
                agent.target_net.state_dict(),

            "epsilon":
                agent.epsilon

        }, save_path)

        checkpoint_pool.append(save_path)

        # Keep pool size limited
        if len(checkpoint_pool) > MAX_CHECKPOINT_POOL:

            checkpoint_pool.pop(0)

        print(f"Checkpoint saved: {save_path}")
        
    # =========================
    # Update frozen opponent
    # =========================
    if episode > 0 and episode % 500 == 0:

        print(f"Updating self-play opponent at episode {episode}")

        agent.update_opponent()

    # =========================
    # Evaluation
    # =========================
    if episode % EVAL_INTERVAL == 0:
        avg_checkpoint_win = np.nan
        avg_reward = np.mean(reward_window)
        reward_window = []

        # Heuristic eval
        heuristic_results = evaluate_against_opponent(
            agent,
            env,
            opponent_type="heuristic",
            n_games=EVAL_GAMES
        )
        
        h_win = heuristic_results["win_rate"]
        h_loss = heuristic_results["loss_rate"]
        h_draw = heuristic_results["draw_rate"]
        
        # Random eval
        random_results = evaluate_against_opponent(
            agent,
            env,
            opponent_type="random",
            n_games=EVAL_GAMES
        )
        
        r_win = random_results["win_rate"]
        r_loss = random_results["loss_rate"]
        r_draw = random_results["draw_rate"]

        # =================================================
        # Historical Checkpoint Evaluation
        # =================================================
        if len(checkpoint_pool) > 1:
        
            # Exclude newest checkpoint
            eval_pool = checkpoint_pool[:-1]
        
            checkpoint_eval_win_rates = []
        
            for checkpoint in eval_pool:
        
                results = evaluate_against_checkpoint(
                    agent,
                    env,
                    checkpoint,
                    n_games= 10
                )
        
                checkpoint_eval_win_rates.append(
                    results["win_rate"]
                )
        
                print(
                    f"{os.path.basename(checkpoint)} | "
                    f"W:{results['wins']} "
                    f"L:{results['losses']} "
                    f"D:{results['draws']} "
                    f"WR:{results['win_rate']:.3f}"
                )
        
            avg_checkpoint_win = np.mean(
            checkpoint_eval_win_rates
        )
        
            print(
                f"Checkpoint Avg Win Rate: "
                f"{avg_checkpoint_win:.3f}"
            )
        
            print(
                f"Individual Checkpoints: "
                f"{[round(x, 3) for x in checkpoint_eval_win_rates]}"
            )

        # Logging
        print(
            f"Episode {episode} | "
            f"Heuristic Win: {h_win:.2f} | "
            f"Random Win: {r_win:.2f} | "
            f"Eps: {agent.epsilon:.3f}"
        )

        heuristic_win_rates.append(h_win)
        random_win_rates.append(r_win)

        loss_rates.append(h_loss)
        draw_rates.append(h_draw)

        heuristic_eval_episodes.append(episode)
        
        training_logs.append({

        "episode": episode,
        "epsilon": agent.epsilon,
        "avg_reward": avg_reward,
        "heuristic_win_rate": h_win,
        "random_win_rate": r_win,
        "avg_checkpoint_win_rate": avg_checkpoint_win,

        })
        
training_df = pd.DataFrame(training_logs)
