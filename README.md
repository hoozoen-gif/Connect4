# Connect 4 Reinforcement Learning Agent

A Deep Q-Network (DQN) based Connect 4 AI built with PyTorch.

This project explores reinforcement learning techniques for board-game AI, including:

* Deep Q-Learning
* Convolutional Neural Networks (CNN)
* Experience Replay
* Target Networks
* Curriculum Learning
* Self-Play Training
* Reward Shaping

The goal is to train an agent that can learn to play Connect 4 from experience and progressively improve against stronger opponents.
## Software Stack

This project is built entirely in Python and uses the PyTorch deep learning framework to train a Reinforcement Learning (RL) agent capable of playing Connect 4.

### Core Technologies

* **Python 3** – Primary programming language
* **PyTorch** – Deep learning framework used to build and train the Deep Q-Network (DQN)
* **NumPy** – Efficient numerical and array operations
* **Tkinter** – Graphical User Interface (GUI) for playing against trained agents
* **Matplotlib** *(optional for evaluation)* – Visualisation of training performance and evaluation metrics

### Reinforcement Learning Components

The project implements a custom RL training pipeline including:

* Deep Q-Network (DQN)
* Experience Replay Buffer
* Epsilon-Greedy Exploration
* Curriculum Learning
* Self-Play Training
* Convolutional Neural Network (CNN) State Encoder

### Environment

Instead of relying on OpenAI Gym, this project uses a fully custom Connect 4 environment built specifically for reinforcement learning experimentation. The environment handles:

* State representation
* Action validation
* Reward calculation
* Win/loss detection
* Opponent interaction
* Training and evaluation workflows
Play Against the AI

### How to play!
A pretrained model is included in this repository so you can immediately play against the current strongest Connect 4 agent without running any training.

Included Model
models/current_best_model.pth

This checkpoint represents the strongest model available at the time of publishing and is used by the gameplay interface.

Launch the Game

Run:

python play_game.py

A Tkinter window will open where:

Red pieces = Human Player
Yellow pieces = AI Agent
Click a column to place your piece
The AI will automatically respond using the pretrained model
Requirements

Install the required dependencies:

pip install torch numpy

Tkinter is included with most standard Python installations.

Notes

The gameplay interface automatically loads:

models/current_best_model.pth

No additional training or configuration is required.
---

## Project Overview

Connect 4 is a classic two-player strategy game played on a 6×7 board. Players alternate dropping pieces into columns, attempting to connect four pieces horizontally, vertically, or diagonally.

This project implements:

* A custom Connect 4 environment
* A DQN agent using PyTorch
* Random and heuristic opponents
* Self-play training
* Evaluation and visualization tools
* Interactive gameplay against trained models

---

## Project Structure

```text
Connect4/
│
├── Connect4Game.py      # Core game logic
├── Connect4Env.py       # Reinforcement learning environment
├── DQN_Agent.py         # DQN implementation
├── Replay_Buffer.py     # Experience replay memory
├── model.py             # CNN model architecture
├── train.py             # Training script
├── play_game.py         # Play against trained model
└── models/              # Saved model checkpoints
```

---

## State Representation

The board is represented using a multi-channel encoding:

### Channel 1

Current player's pieces

### Channel 2

Opponent's pieces

This representation allows the CNN to distinguish between friendly and enemy positions more effectively than a single-channel board representation.

---

## Neural Network Architecture

The agent uses a Convolutional Neural Network:

```text
Input (2 × 6 × 7)

↓ Conv Layers

↓ Feature Extraction

↓ Fully Connected Layers

↓ Q-values (7 actions)
```

The output consists of Q-values for each column:

```text
Action Space = 7 columns
```

The agent selects moves using an ε-greedy policy.

---

## Training Strategy

Training uses a curriculum-learning approach:

### Phase 1

Episodes 0–2000

* 70% Random Opponent
* 30% Heuristic Opponent

Goal:

* Learn legal moves
* Learn basic winning patterns

---

### Phase 2

Episodes 2000–5000

* 40% Random Opponent
* 40% Heuristic Opponent
* 20% Self-Play

Goal:

* Improve tactical awareness
* Begin adapting against stronger opponents

---

### Phase 3

Episodes 5000+

* 10% Random Opponent
* 20% Heuristic Opponent
* 70% Self-Play

Goal:

* Develop advanced strategies
* Improve long-term planning

---

## Reward Shaping

The environment uses reward shaping to accelerate learning.

Examples include:

* Winning a game
* Blocking an opponent's threat
* Creating multiple threats
* Penalizing losing moves
* Discouraging invalid actions

Reward shaping helps the agent discover useful strategies much faster than sparse win/loss rewards alone.

---

## Experience Replay

A replay buffer stores transitions:

```python
(state, action, reward, next_state, done)
```

Mini-batches are randomly sampled during training to:

* Reduce correlation between experiences
* Improve stability
* Increase sample efficiency

---

## Self-Play

After the agent learns basic gameplay, it begins training against previous versions of itself.

Benefits:

* Continuously increasing difficulty
* Reduced dependence on handcrafted heuristics
* Discovery of stronger strategies

---

## Results

Example evaluation metrics:

| Opponent  | Win Rate  |
| --------- | --------- |
| Random    | High      |
| Heuristic | Moderate  |
| Self-Play | Improving |

Training performance is monitored throughout learning to track progress against different opponent types.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hoozoen-gif/Connect4.git

cd Connect4
```

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Training

Run training:

```bash
python train.py
```

The model will periodically save checkpoints for later evaluation.

---

## Playing Against the Agent

Run:

```bash
python play_game.py
```

You can play Connect 4 directly against a trained model.

---

## Future Improvements

Potential upgrades:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* Residual CNN Blocks
* AlphaZero-style MCTS
* Distributed Self-Play
* Stronger heuristic opponents

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Tkinter
* Reinforcement Learning

---

## Learning Objectives

This project was built to explore:

* Reinforcement Learning
* Deep Q-Networks
* CNNs for board games
* Self-play systems
* Curriculum learning
* Game AI development

---

## License

This project is released under the MIT License.

```
```
