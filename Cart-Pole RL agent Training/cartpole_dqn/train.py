"""Train a DQN agent on CartPole-v1."""
import argparse

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent

ENV_ID = "CartPole-v1"
MAX_STEPS = 500
SOLVE_SCORE = 475.0
SOLVE_WINDOW = 100


def train(num_episodes: int = 400, quiet: bool = False):
    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        avg_last_100 = np.mean(rewards_history[-SOLVE_WINDOW:])
        if not quiet and episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward {episode_reward:6.1f} | "
                  f"Avg(100) {avg_last_100:6.1f} | Epsilon {agent.eps:.3f}")

        if episode >= SOLVE_WINDOW and avg_last_100 >= SOLVE_SCORE:
            print(f"Solved at episode {episode}! Avg(100) = {avg_last_100:.1f}")
            break

    agent.save("cartpole_dqn.pt")
    env.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rewards_history, alpha=0.4, label="Episode reward")
    if len(rewards_history) >= SOLVE_WINDOW:
        moving_avg = np.convolve(rewards_history, np.ones(SOLVE_WINDOW) / SOLVE_WINDOW, mode="valid")
        plt.plot(range(SOLVE_WINDOW - 1, len(rewards_history)), moving_avg, label="Avg(100)")
    plt.axhline(SOLVE_SCORE, color="red", linestyle="--", label="Solve threshold")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole-v1 DQN Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cartpole_training_curve.png", dpi=150)
    print("Saved model to cartpole_dqn.pt and curve to cartpole_training_curve.png")
    return rewards_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400, help="Max training episodes")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode logging")
    args = parser.parse_args()
    train(num_episodes=args.episodes, quiet=args.quiet)
