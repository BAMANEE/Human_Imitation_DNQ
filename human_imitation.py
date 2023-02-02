"""
Imitation learning using a neural network to learn from human demonstrations.
"""
import pickle
import argparse
import gym
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from model import ImitationNetwork, ImitationNetworkImage
from game_params import game_params

LR = 1e-4

class ExperienceTrainDataset(Dataset):
    """
    Dataset for training the imitation learning model.
    """
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def get_labels(self):
        return self.actions

def create_validation_split(data, split_ratio):
    """
    Splits the data into a training and validation set.
    """
    data_x = [experience[0] for experience in data]
    data_y = [experience[1] for experience in data]
    split = int(len(data_x) * split_ratio)
    return np.array(data_x[:split]), np.array(data_y[:split]), np.array(data_x[split:]), np.array(data_y[split:])


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch on the given data.
    """
    model.train()
    optimizer.zero_grad()
    loss = 0
    losses = []

    for _, sample in enumerate(loader):

        state = sample[0].to(device)
        action = sample[1].to(device)

        output = model(state)
        loss = criterion(output, action)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, criterion, device):
    """
    Validates the model on the given data.
    """
    model.eval()
    losses = []
    for i, sample in enumerate(loader):
        state = sample[0].to(device)
        action = sample[1].to(device)

        output = model(state)
        loss = criterion(output, action)
        losses.append(loss.item())
    return np.mean(losses)

def accuracy(model, loader, device):
    """
    Calculates the accuracy of the model on the given data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            state = sample[0].to(device)
            action = sample[1].to(device)

            output = model(state)
            _, predicted = torch.max(output.data, 1)
            total += action.size(0)
            correct += (predicted == action).sum().item()
    return correct / total
    
def train_and_validate(data_x, data_y, valid_x, valid_y, epochs, output, image=False):
    """
    Trains the model on the given data and saves the best model to the given output path.
    """

    if image:
        model = ImitationNetworkImage(state_size, action_size, seed).to(device)
        batch_size = 64
    else:    
        model = ImitationNetwork(state_size, action_size, seed).to(device)
        batch_size = 1024

    train_dataset = ExperienceTrainDataset(data_x, data_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_dataset))
    valid_dataset = ExperienceTrainDataset(valid_x, valid_y)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(valid_dataset))

    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lrr = 0.1**(1/epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lrr ** epoch)
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = (0, np.inf)
    for epoch in range(1, epochs+1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = validate(model, valid_loader, criterion, device)
        acc = accuracy(model, valid_loader, device)
        scheduler.step()
        print(f'Epoch: {epoch}\tLoss: {loss:.4f}\tValidation loss: {valid_loss:.4f}\tAccuracy: {acc:.4f}')
        torch.save(model.state_dict(), output + "Latest.pth")
        if valid_loss < best_valid_loss[1]:
            best_valid_loss = (epoch, valid_loss)
            torch.save(model.state_dict(), output + "Best.pth")
    model.load_state_dict(torch.load(args.output + "Best.pth"))
    print(f'Lowest validation loss {best_valid_loss[1]:.4f} after {best_valid_loss[0]} epochs with a accuracy of {accuracy(model, valid_loader, device):.4f}')
    return model


def demo(model, env, device, episodes=1000):
    """
    Runs the model on the given environment and prints the average score.
    """
    scores = []
    for _ in tqdm(range(episodes)):
        score = 0
        state = env.reset()
        for j in range(1000):
            action = torch.argmax(model(torch.from_numpy(state).type(torch.float).to(device))).item()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
            scores.append(score)
    print(f"Average score: {np.mean(scores):.2f}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Imitation Learning')
    parser.add_argument('experiences', type=str, help='data file')
    parser.add_argument('game', type=str, help='game name')
    parser.add_argument('epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('output', type=str, help='output file')
    parser.add_argument('--image', action='store_true', help='use image as input')
    args = parser.parse_args()

    experiences = pickle.load(open(args.experiences, "rb"))
    data_x, data_y, valid_x, valid_y = create_validation_split(experiences, 0.8)
    data_x = torch.from_numpy(data_x).type(torch.float)
    data_y = torch.from_numpy(data_y).type(torch.LongTensor)
    valid_x = torch.from_numpy(valid_x).type(torch.float)
    valid_y = torch.from_numpy(valid_y).type(torch.LongTensor)

    params = game_params.get(args.game)
    if not params:
        raise ValueError(f"Invalid game name: {args.game}")

    state_size = params["state_size"]
    action_size = params["action_size"]
    max_t = params["max_t"]

    if args.game.startswith("ALE/"):
        if args.image:
            env = gym.make(args.game, obs_type="rgb", full_action_space=False)
        else:
            env = gym.make(args.game, obs_type="ram", full_action_space=False)
    else:
        env = gym.make(args.game)


    seed = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = train_and_validate(data_x, data_y, valid_x, valid_y, epochs=args.epochs, output=args.output, image=args.image)

    demo(model, env, device, episodes=1000)