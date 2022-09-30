import gym
from HumanActionGetter import *
from pynput.keyboard import Key, Listener
import argparse
import numpy as np
import random

class InputState:
    def __init__(self):
        self.pressed_keys = set()

def keypress_callback(inputState, key):
        inputState.pressed_keys.add(key)

def keyrelease_callback(inputState, key):
        inputState.pressed_keys.remove(key)

def collect_human_experiences(game, output, n_episodes, max_t=2000):
    inputState = InputState()

    if game == "MountainCar-v0":
        action_getter =  MountainCarHumanActionGetter()
        env = gym.make(game)
    elif game == "LunarLander-v2":
        action_getter = LunarLanderHumanActionGetter()
        env = gym.make(game, render_mode="human")
    elif game == "CartPole-v1":
        action_getter = CartPoleHumanActionGetter()
        env = gym.make(game, render_mode="human")
    elif game == "ALE/Breakout-v5":
        action_getter = BreakoutHumanActionGetter()
        env = gym.make(game, render_mode="human", full_action_space=False, obs_type="ram")
    else:  
        raise Exception("Unknown game: {}".format(game))

    experiences = []
    scores = []
    with Listener(on_press=lambda key: keypress_callback(inputState, key), on_release=lambda key: keyrelease_callback(inputState, key)) as listener:
        for i in range(n_episodes):
            state = env.reset()
            score = 0
            for t in range(max_t):
                #action = action_getter.get_action(inputState)
                action = random.randint(0, 2)
                next_state, reward, done, _ = env.step(action)
                score += reward
                experiences.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    break
            print("episode: {} score: {}".format(i, score))
            scores.append(score)
    
    print("Average score: {:.2f}".format(np.mean(scores)))

    env.close()
    pickle.dump(experiences, open(output, "wb"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect human experiences')
    parser.add_argument('game', type=str, help='game to collect experiences for')
    parser.add_argument('output', type=str, help='output file')
    parser.add_argument('n_episodes', type=int, help='number of episodes to collect')
    args = parser.parse_args()
    collect_human_experiences(args.game, args.output, args.n_episodes)


