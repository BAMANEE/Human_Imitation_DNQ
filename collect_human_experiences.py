"""
Collect human experiences for a given game.
"""
import argparse
import time
from pynput.keyboard import Listener
import numpy as np
import gym

from HumanActionGetter import *
from game_params import *

class InputState:
    """Class to store the state of the keyboard."""	
    def __init__(self):
        self.pressed_keys = set()

def keypress_callback(input_state, key):
    """Callback function for keypresses."""
    input_state.pressed_keys.add(key)

def keyrelease_callback(input_state, key):
    """Callback function for keyreleases."""
    input_state.pressed_keys.remove(key)

def collect_human_experiences(game, output, n_episodes, max_t=2000, image=False):
    """Collect human experiences for a given game."""	
    input_state = InputState()
    frame_pause = 0

    params = game_params[game]
    if not params:
        raise ValueError(f"Unknown game: {game}")
    
    action_getter = params["action_getter"]()

    if game.startswith("ALE"):
        if image:
            env = gym.make(game, render_mode="human", full_action_space=False, obs_type="rgb")
        else:
            env = gym.make(game, render_mode="human", full_action_space=False, obs_type="ram")
    else:
        env = gym.make(game, render_mode="human")

    experiences = []
    scores = []
    with Listener(on_press=lambda key: keypress_callback(input_state, key), on_release=lambda key: keyrelease_callback(input_state, key)) as _:
        for i in range(n_episodes):
            state = env.reset()
            score = 0
            for _ in range(max_t):
                action = action_getter.get_action(input_state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                experiences.append((state, action, reward, next_state, done))
                state = next_state
                time.sleep(frame_pause)
                if done:
                    break
            print(f"episode: {i} score: {score}")
            scores.append(score)

    print(f"Average score: {np.mean(scores):.2f}")

    env.close()
    pickle.dump(experiences, open(output, "wb"))   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect human experiences')
    parser.add_argument('--game', type=str, help='game to collect experiences for')
    parser.add_argument('--output', type=str, help='output file')
    parser.add_argument('--n_episodes', type=int, help='number of episodes to collect')
    parser.add_argument("--image", action="store_true", help="collect images instead of states")
    args = parser.parse_args()
    collect_human_experiences(args.game, args.output, args.n_episodes, args.image)