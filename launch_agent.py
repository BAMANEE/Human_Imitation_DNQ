from agent import Agent
import gym
import torch
import random 
import pickle
import numpy as np

env = gym.make("MountainCar-v0")
env.seed(0)
agent = Agent(state_size=2, action_size=3, seed=0)
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('dqnMountainCarModel.pt'))

experiences = []
scores = []

for i in range (10000):
        state = env.reset()
        score = 0
        for j in range(200):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            experiences.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                break
        scores.append(score)
        print("episode: {} score: {}".format(i, score), end="\r")
print("Average score: {}".format(np.mean(scores)))
            
pickle.dump(experiences, open("Ai10000.pkl", "wb"))