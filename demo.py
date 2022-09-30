import gym
import torch
from model import ImitationNetwork
import numpy as np
    
def demo(model, env, device):
    scores = []
    for j in range (1000):
            score = 0
            state = env.reset()
            for j in range(200):
                action = torch.argmax(model(torch.from_numpy(state).type(torch.float).to(device))).item()
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break 
            scores.append(score)
    print("Average score: {:.2f}".format(np.mean(scores)))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make('MountainCar-v0')
    state_size = 2
    action_size = 3
    seed = 0
    model = ImitationNetwork(state_size, action_size, seed).to(device)
    model.load_state_dict(torch.load("imitation_models/GoodHuman100ModelLatest.pth"))
    demo(model, env, device)