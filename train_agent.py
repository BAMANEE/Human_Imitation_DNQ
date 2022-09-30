import os
import gym
import random
import torch
import numpy as np
from collections import deque
from collections import namedtuple, deque
import random
import time
from agent import Agent
from model import ImitationNetwork
import argparse

def dqn(game, n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.001, eps_decay=0.995, threshold=200, eps_deg_start = 0, second_model=None, extra_rand=0):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        threshold (float): score above which the agent is considered to have succeeded
        eps_deg_start (int): number of episodes after which epsilon starts to decay
        second_model (nn.Module): second model to be used for imitation learning
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    start_time = time.time()
    env = gym.make(game)
    scores_window = deque(maxlen=100) 
    print("\nStart computer training")
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps, second_model, extra_rand)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        if i_episode >= eps_deg_start:
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode: {}\tAverage Score: {:.2f}\tEpsilon: {:.4f}\tDelta: {:.4f}'.format(i_episode, np.mean(scores_window), eps, agent.get_delta()), end="")
        if i_episode % 100 == 0:
            print('\rEpisode: {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            elapsed_time = time.time() - start_time
            print("Duration: ", elapsed_time)
        if (i_episode >= 100 and np.mean(scores_window)>=threshold):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
        if i_episode == n_episodes:
            print('\nEnvironment not solved after {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, help='Name of the game')
    parser.add_argument('--n_episodes', type=int, help='Number of episodes')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='Epsilon decay')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--second_model', type=str, default=None, help='Path to second model')
    parser.add_argument('--results_folder', type=str, default=None, help='Path to results folder')
    parser.add_argument('--model_save_path', type=str, default=None, help='Path to model save folder')
    parser.add_argument('--extra_rand', type=float, default=0, help='Extra random actions')

    args = parser.parse_args()

    if args.game == "MountainCar-v0":
        state_size = 2
        action_size = 3
        threshold = -90
    
    for run in range(1, args.num_runs+1):
        print("Starting run ", run)
        agent = Agent(state_size=state_size, action_size=action_size, seed=run)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.second_model:
            human_model = ImitationNetwork(state_size=state_size, action_size=action_size, seed=run).to(device)
            human_model.load_state_dict(torch.load(args.second_model))
            human_model.eval()
        else:
            human_model = None
        scores = dqn(args.game, n_episodes=args.n_episodes ,eps_start=1, eps_decay=args.eps_decay, threshold=threshold, second_model=human_model, extra_rand=args.extra_rand)
        if args.results_folder:
            if not os.path.exists(args.results_folder):
                os.makedirs(args.results_folder)
            with open(args.results_folder + "/" + str(run) + ".txt", 'w') as f:
                for score in scores:
                    f.write(str(score) + "\n")
        if args.model_save_path:
            torch.save(agent.qnetwork_local.state_dict(), args.model_save_path + str(run) + ".pt")