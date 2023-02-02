"""
Launch an agent in a game and record the experiences
"""
import pickle
import argparse
import time
import gym
import torch
import numpy as np

from agent import Agent
from game_params import game_params

def launch_agent(game, weights_path, n_episodes, max_t = 2000, experience_output = None, render=False, image=False):
        """
        Launch an agent in a game and record the experiences
        """
        params = game_params.get(game)
        if not params:
                raise ValueError(f"Unknown game: {game}")
        
        state_size = params["state_size"]
        action_size = params["action_size"]

        if render:
                render_mode = "human"
        else:
                render_mode = "None"

        if game.startswith("ALE"):
                if image:
                        env = gym.make(game, render_mode=render_mode, full_action_space=False, obs_type="rgb")
                else:
                        env = gym.make(game, render_mode=render_mode, full_action_space=False, obs_type="ram")
        else:
                env = gym.make(game, render_mode=render_mode)

        env.seed(0)
        agent = Agent(state_size, action_size, seed=0)
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load(weights_path))

        experiences = []
        scores = []

        for i in range (n_episodes):
                state = env.reset()
                score = 0
                for j in range(max_t):
                        action = agent.act(state)
                        next_state, reward, done, _ = env.step(action)
                        score += reward
                        experiences.append((state, action, reward, next_state, done))
                        state = next_state
                        if game == "FlappyBird-v0" and render:
                                env.render()
                                time.sleep(1 / 30)  # FPS
                        if done:
                                break
                scores.append(score)
                print(f"episode: {i} score: {score}", end="\r")
        print(f"Average score: {np.mean(scores)}")

        if experience_output is not None:
                with open(experience_output, "wb") as f:
                        pickle.dump(experiences, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning')
    parser.add_argument('game', type=str, help='game name')
    parser.add_argument('weigths', type=str, help='weights file')
    parser.add_argument('n_episodes', type=int, default=10, help='number of epochs')
    parser.add_argument('--experience_output', type=str, default=None, help='output file for experiences')
    parser.add_argument('--render', type=bool, default=False, help='render the game')
    parser.add_argument('--image', action='store_true', help='use image as input')
    
    args = parser.parse_args()

    launch_agent(args.game, args.weigths, args.n_episodes, experience_output=args.experience_output, render=args.render, image=args.image)