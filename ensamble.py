import numpy as np
import torch
from model import *
import gym
import argparse
from tqdm import trange

from game_params import game_params


def majority_voting(models, state):
    actions = []
    for model in models:
        action = model(state).argmax().item()
        actions.append(action)
    return max(set(actions), key=actions.count)

def average_voting(models, state):
    predictions = []
    for model in models:
        prediction = model(state)
        predictions.append(prediction)
    predictions = torch.stack(predictions)
    return torch.mean(predictions, dim=0).argmax().item()

def random_voting(models, state):
    model = np.random.choice(models)
    return model(state).argmax().item()

def ensamble(models, state, ensamble_type="average"):
    if ensamble_type == "majority":
        return majority_voting(models, state)
    elif ensamble_type == "average":
        return average_voting(models, state)
    elif ensamble_type == "random":
        return random_voting(models, state)
    else:
        raise ValueError("Invalid ensamble type.")

def ensamble_run(models, env, episodes, max_t, device, ensamble_type="majority"):
    scores = []
    for j in trange (episodes):
            score = 0
            state = env.reset()
            for j in range(max_t):
                state = torch.from_numpy(state).type(torch.float).to(device)
                action = ensamble(models, state, ensamble_type=ensamble_type)
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
            scores.append(score)
    print("Average score: {:.2f}".format(np.mean(scores)))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True, help="The game to test on.")
    parser.add_argument("--episodes", type=int, default=100, help="The number of episodes to test.")
    parser.add_argument("--models", type=str, nargs='+', help="The model files to test.")	
    parser.add_argument("--ensamble_type", type=str, default="majority", help="The ensamble type to use.")
    parser.add_argument("--display", action="store_true", help="Display the game.")
    args = parser.parse_args()    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    game_params = game_params[args.game]
    if args.display:
        env = gym.make(args.game, render_mode="human")
    else:
        env = gym.make(args.game, )
    state_size = game_params["state_size"]
    action_size = game_params["action_size"]
    max_t = game_params["max_t"]
    seed = 0
    models = []
    for model_file in args.models:
        model = ImitationNetwork(state_size, action_size, seed).to(device)
        model.load_state_dict(torch.load(model_file))
        models.append(model)
    ensamble_run(models, env, args.episodes, max_t, device, ensamble_type=args.ensamble_type)
    