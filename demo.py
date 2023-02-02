import gym
import torch
from model import *
import numpy as np
from tqdm import trange
    
def demo(model, env, device):
    scores = []
    for j in trange (100):
            score = 0
            state = env.reset()
            for j in range(200):
                if image == True:
                    #expand dimension to fit model
                    state = np.expand_dims(state, axis=0)
                action = torch.argmax(model(torch.from_numpy(state).type(torch.float).to(device))).item()
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break 
            scores.append(score)
    print("Average score: {:.2f}".format(np.mean(scores)))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('CarRacing-v2', continuous=False)
    state_size = (92, 92, 3)
    action_size = 5
    seed = 0
    image = True
    if image == True:
        model = ImitationNetworkImage(state_size, action_size, seed).to(device)
    else:
        model = ImitationNetwork(state_size, action_size, seed).to(device)
    model.load_state_dict(torch.load("imitation_models/CarRacing10Best.pth"))
    demo(model, env, device)