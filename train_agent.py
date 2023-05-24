"""
Train an agent to play a game using DQN and imitation learning.
"""

import os
import sys
import time
from collections import deque
import argparse
import numpy as np
from tqdm import tqdm, trange
import torch
import gym

from agent import Agent
from model import ImitationNetwork, ImitationNetworkImage
from game_params import game_params

STOP_AFTER_REACHED_THRESHOLD = False


def dqn(env, n_episodes=2000, max_t=2000, psi_start=1, psi_end=0.0, psi_decay=0.995, psi_deg_start=0, eps_start=0, eps_end=0, eps_decay=1, threshold=200, second_model=None, extra_rand=0, buffer_size=10000, image=False):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        psi_start (float): starting value of psi
        psi_end (float): minimum value of psi
        psi_decay (float): multiplicative factor (per episode) for decreasing psi
        threshold (float): score above which the agent is considered to have succeeded
        second_model (nn.Module): second model to be used for imitation learning
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    psi = psi_start                    # initialize psi
    eps = eps_start                    # initialize epsilon

    start_time = time.time()

    desc = '\rAverage Score: \tPsi: \tDelta: '
    pbar = tqdm(trange(1, n_episodes+1), position=0, leave=True)
    for i_episode in pbar:
        pbar.set_description(desc)
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action = agent.act(state, psi, eps, second_model)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
             
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        desc = f'\rAverage Score: {np.mean(scores_window):.2f}, psi: {psi:.2f}, eps: {eps:.4f}'
        if i_episode >= psi_deg_start:
            psi = max(psi_end, psi_decay*psi) # decrease psi
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        if (i_episode >= 100 and STOP_AFTER_REACHED_THRESHOLD and np.mean(scores_window)>=threshold):
            print(f'\nEnvironment solved in {i_episode-100} episodes!, Average Score: {np.mean(scores_window):.2f}')
            break
        if i_episode == n_episodes:
            print(f'\nEnvironment not solved after {i_episode} episodes!, Average Score: {np.mean(scores_window):.2f}')
            break
    elapsed_time = time.time() - start_time
    print("Training duration: ", elapsed_time)
    return scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, help='Name of the game')
    parser.add_argument('--n_episodes', type=int, help='Number of episodes')
    parser.add_argument('--psi_start', type=float, default=1, help='Psi start')
    parser.add_argument('--psi_decay', type=float, default=0.995, help='Psi decay')
    parser.add_argument('--psi_end', type=float, default=0.01, help='Psi end')
    parser.add_argument('--psi_deg_start', type=int, default=0, help='Psi decay start')
    parser.add_argument('--eps_start', type=float, default=1, help='Epsilon start')	
    parser.add_argument('--eps_end', type=float, default=0, help='Epsilon end')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='Epsilon decay')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--first_model', type=str, default=None, help='Path to first model')
    parser.add_argument('--finetune', action='store_true', help='Fine tune model')
    parser.add_argument('--second_model', type=str, nargs='+', default=None, help='Path to second model')
    parser.add_argument('--results_folder', type=str, default=None, help='Path to results folder')
    parser.add_argument('--model_save_path', type=str, default=None, help='Path to model save folder')
    parser.add_argument('--extra_rand', type=float, default=0, help='Extra random actions')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--image', action='store_true', help='use image as input')
    parser.add_argument('--ensamble_method', type=str, default='average', help='Ensamble method')
    args = parser.parse_args()

    start_run = 1

    if args.results_folder:
        if not os.path.exists(args.results_folder):
            os.makedirs(args.results_folder)
            with open(os.path.join(args.results_folder, "parameters.txt"), "w", encoding="utf-8") as f:
                f.write(str(args))
        else:
            with open(os.path.join(args.results_folder, "parameters.txt"), "r", encoding="utf-8") as f:
                parameters = f.read()
                #if parameters == f"Game: {args.game}\tNumber of episodes: {args.n_episodes}\tPsi decay: {args.psi_decay}\tNumber of runs: {args.num_runs}\tExtra random actions: {args.extra_rand}\tSecond model: {args.second_model}\tModel save path: {args.model_save_path}\tBuffer size: {args.buffer_size}":
                start_run = len([entry for entry in os.listdir(args.results_folder) if os.path.isfile(os.path.join(args.results_folder, entry))])
                """"
                else:
                    print("Parameters do not match!")
                    sys.exit() 
                """

    params = game_params.get(args.game)
    if not params:
        raise ValueError(f"Invalid game name: {args.game}")

    state_size = params["state_size"]
    action_size = params["action_size"]
    threshold = params["threshold"]
    max_t = params["max_t"]

    if args.game.startswith("ALE/"):
        if args.image:
            env = gym.make(args.game, obs_type="image", full_action_space=False)
        else:
            env = gym.make(args.game, obs_type="ram", full_action_space=False)
    else:
        env = gym.make(args.game)

    
    for run in range(start_run, args.num_runs+1):
        print("Starting run ", run)
        agent = Agent(state_size=state_size, action_size=action_size, seed=run, buffer_size=args.buffer_size, img=args.image, finetune=args.finetune, ensamble_method=args.ensamble_method)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.first_model:
            agent.qnetwork_local.load_state_dict(torch.load(args.first_model))
            if args.finetune:
                #freeze first layers
                agent.qnetwork_local.fc1.requires_grad_(False)
                agent.qnetwork_local.fc2.requires_grad_(False)
        if args.second_model:
            psi_start = args.psi_start
            psi_end = args.psi_end
            human_models = []
            for model_files in args.second_model:
                if args.image:
                    human_model = ImitationNetworkImage(state_size=state_size, action_size=action_size, seed=run).to(device)
                else:
                    human_model= ImitationNetwork(state_size=state_size, action_size=action_size, seed=run).to(device)
                human_model.load_state_dict(torch.load(model_files))
                human_model.eval()
                human_models.append(human_model)
        else:
            psi_start = 0
            psi_end = 0
            human_models = None

        results = dqn(env, n_episodes=args.n_episodes, max_t=max_t, psi_start=psi_start, psi_decay=args.psi_decay, psi_end=psi_end, psi_deg_start=args.psi_deg_start,
                      eps_start=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay, threshold=threshold, second_model=human_models, extra_rand=args.extra_rand,
                      image=args.image)
        if args.results_folder:
            with open(args.results_folder + "/" + str(run) + ".txt", 'w', encoding="utf-8") as f:
                for result in results:
                    f.write(str(result) + "\n")
        if args.model_save_path:
            torch.save(agent.qnetwork_local.state_dict(), args.model_save_path + "_run_" + str(run) + ".pt")