# Lunar Lander PPO

Trains a Proximal Policy Optimization (PPO) agent, via Stable-Baselines3, to
land the craft in the `LunarLander-v3` environment (Gymnasium/Box2D).

## Project structure
```
lunarlander_ppo/
├── requirements.txt   # dependencies
├── train.py             # trains PPO with 8 parallel envs, saves best model + logs
├── evaluate.py          # loads a trained model and runs it greedily
└── README.md
```

## Setup
```bash
# System dependency needed to build the box2d physics engine
sudo apt-get install swig        # Debian/Ubuntu
brew install swig                # macOS

python3 -m venv venv
source venv/bin/activate         # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
python train.py                          # default: 1,000,000 timesteps, 8 parallel envs
python train.py --timesteps 300000 --n-envs 4   # shorter run on fewer cores
```
This saves:
- `lunarlander_ppo.zip` — final trained policy
- `logs/best_model.zip` — best policy seen during evaluation callbacks
- `logs/` — TensorBoard logs and evaluation history

Monitor training live:
```bash
tensorboard --logdir logs
```

## Evaluate
```bash
python evaluate.py                       # 100 greedy episodes, no rendering
python evaluate.py --render               # watch it land
python evaluate.py --model logs/best_model --episodes 20 --render
```

## Algorithm notes
- **Method:** PPO (clipped surrogate objective) via Stable-Baselines3.
- **Parallelism:** 8 vectorized environments for stable, faster data collection.
- **Hyperparameters:** tuned defaults known to work well on LunarLander
  (gamma=0.999, gae_lambda=0.98, ent_coef=0.01, n_steps=1024).
- **Solved criterion:** average reward ≥ ~200 over 100 consecutive episodes.

## Suggested extensions
- Try `LunarLanderContinuous-v3` with PPO or SAC for continuous thrust control.
- Hyperparameter sweep with Optuna's `BaseTrialCallback`.
- Record an MP4 with `gymnasium.wrappers.RecordVideo`.
