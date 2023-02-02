"""
Test the performance of a model on a given environment.
"""

import argparse
import os
import gym
import torch
from tqdm import trange
import numpy as np

from agent import Agent

def test_model_performance(agent, env, episodes):
    """Test the performance of a model on a given environment.
    Args:
        model: The model to test.
        env: The environment to test on.
        runs: The number of runs to test.
    Returns:
        The average reward over the runs.
    """
    scores = []
    for _ in trange(episodes):
        score = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            score += reward
        scores.append(score)        
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return mean_score, std_score

def average_model_performance(agent, model_files, env, episodes):
    """Test the performance of a model on a given environment.
    Args:
        model: The model to test.
        env: The environment to test on.
        runs: The number of runs to test.
    Returns:
        The average reward over the runs.
    """
    scores = []
    stds = []
    for model_file in model_files:
        agent.qnetwork_local.load_state_dict(torch.load(model_file))
        mean_score, std_score = test_model_performance(agent, env, episodes)
        print(f"Model: {model_file} Average reward: {mean_score:.2f} +/- {std_score:.2f}")
        scores.append(mean_score)
        stds.append(std_score)
    mean_scores = np.mean(scores)
    std_scores = np.mean(stds)
    return mean_scores, std_scores

def collect_model_files(model_files):
    path, filename = os.path.split(model_files)
    files = []
    for file in os.listdir(path):
        if file.startswith(filename):
            files.append(os.path.join(path, file))
    return files
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True, help="The game to test on.")
    parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to test.")
    parser.add_argument("--model_files", type=str, help="The model files to test.")	
    args = parser.parse_args()

    if args.game == "MountainCar-v0":
        state_size = 2
        action_size = 3
        seed = 0
        env = gym.make('MountainCar-v0')
    agent = Agent(state_size, action_size, seed=seed)
    model_files = collect_model_files(args.model_files)
    mean_scores, std_scores = average_model_performance(agent, model_files, env, args.episodes)
    print(f"Average reward: {mean_scores:.2f} +/- {std_scores:.2f}")
