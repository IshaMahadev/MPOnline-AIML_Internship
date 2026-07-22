# CartPole DQN

A from-scratch PyTorch implementation of Deep Q-Learning (DQN) for the
`CartPole-v1` environment (Gymnasium).

## Project structure
```
cartpole_dqn/
├── requirements.txt   # dependencies
├── dqn_agent.py        # QNetwork, ReplayBuffer, DQNAgent
├── train.py             # training loop, saves cartpole_dqn.pt + a reward plot
├── evaluate.py          # loads a trained model and runs it greedily
└── README.md
```

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
python train.py                 # default: up to 400 episodes
python train.py --episodes 250  # custom episode cap
```
This prints progress every 10 episodes, stops early once the 100-episode
average reward reaches 475, and saves:
- `cartpole_dqn.pt` — trained Q-network weights
- `cartpole_training_curve.png` — reward-vs-episode plot

## Evaluate
```bash
python evaluate.py                      # 100 greedy episodes, no rendering
python evaluate.py --render             # watch it play
python evaluate.py --episodes 20 --render
```

## Algorithm notes
- **Method:** Deep Q-Network (DQN) with a target network and experience replay.
- **Network:** 2 hidden layers (128 units, ReLU).
- **Exploration:** epsilon-greedy, decaying from 1.0 to 0.05.
- **Loss:** Smooth L1 (Huber) between predicted Q-values and TD targets.
- **Solved criterion:** average reward ≥ 475 over 100 consecutive episodes.

## Suggested extensions
- Double DQN (decouple action selection/evaluation to reduce overestimation)
- Dueling network architecture (separate value/advantage streams)
- Prioritized experience replay
