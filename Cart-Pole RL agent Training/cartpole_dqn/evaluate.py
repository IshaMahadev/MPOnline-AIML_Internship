"""Evaluate a trained CartPole DQN agent greedily."""
import argparse

import numpy as np
import gymnasium as gym

from dqn_agent import DQNAgent

ENV_ID = "CartPole-v1"


def evaluate(model_path: str = "cartpole_dqn.pt", episodes: int = 100, render: bool = False):
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)

    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    env.close()
    print(f"Mean reward over {episodes} episodes: "
          f"{np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cartpole_dqn.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(model_path=args.model, episodes=args.episodes, render=args.render)
