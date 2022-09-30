import numpy as np
import random

from model import QNetwork, QnetworkImage
from replay_buffer import PriorityReplayBuffer, ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import operator
from collections import deque

BUFFER_SIZE = int(1e5)      # replay buffer size
BATCH_SIZE = 64             # minibatch size
GAMMA = 0.99                # discount factor
TAU = 1e-3                  # for soft update of target parameters
LR = 5e-4                   # learning rate 
UPDATE_NN_EVERY = 1        # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 20          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000     # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, compute_weights = False, img=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights

        # Q-Network
        if img:
            self.qnetwork_local = QnetworkImage(state_size, action_size, seed).to(device)
            self.qnetwork_target = QnetworkImage(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = PriorityReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, seed, compute_weights)
        self.human_memory = PriorityReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, seed, compute_weights)
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0
        # Initialize time step (for adding human memory every ADD_HUMAN_MEM_EVERY steps)
        self.human = False

        self.delta_window = deque(maxlen=100)

    def pretrain(self, experiences, num_epochs):
        """Pretrain the Q-network with experiences from the replay buffer."""

        

        states = torch.from_numpy(
            np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = [0] * len(experiences)
        indices = [0] * len(experiences)

        experiences = (states, actions, rewards, next_states, dones, weights, indices)

        for epoch in range(num_epochs):
            print("Epoch: {:07d}\tDelta: {:.4f}".format(epoch, self.get_delta()), end='\r')
            self.learn(experiences, GAMMA, pretrain=True)
    
    def step(self, state, action, reward, next_state, done):
        if self.human:
            # Initialize time step (for updating every UPDATE_NN_EVERY steps)
            self.t_step_nn = 0
            # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
            self.t_step_mem_par = 0
            # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
            self.t_step_mem = 0
            # Initialize time step (for adding human memory every ADD_HUMAN_MEM_EVERY steps)
            self.human = False
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > EXPERIENCES_PER_SAMPLING:
                sampling = self.memory.sample()
                self.learn(sampling, GAMMA)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def human_step(self, state, action, reward, next_state, done):
        self.human = True
        # Save experience in replay memory
        self.human_memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % UPDATE_NN_EVERY
        self.t_step_mem = (self.t_step_mem + 1) % UPDATE_MEM_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.human_memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.human_memory.experience_count > EXPERIENCES_PER_SAMPLING:
                sampling = self.human_memory.sample()
                self.learn(sampling, GAMMA)
        if self.t_step_mem == 0:
            self.human_memory.update_memory_sampling()

    def act(self, state, eps=0., other_model=None, extra_rand=0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        rnd = random.random()
        if random.random() > extra_rand:
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            elif other_model is not None:
                return np.argmax(other_model(state).cpu().data.numpy())
        return random.choice(np.arange(self.action_size))

    def learn(self, sampling, gamma, pretrain=False):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices  = sampling

        ## TODO: compute and minimize the loss        
        q_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        expected_values = rewards + gamma*q_target*(1-dones)
        output = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(output, expected_values)
        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update priorities ------------------- #
        delta = abs(expected_values - output.detach()).cpu().numpy()
        self.delta_window.append(delta)
        if self.human:
            self.human_memory.update_priorities(delta, indices)      
        elif not pretrain:
            self.memory.update_priorities(delta, indices)  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def get_delta(self):
        if not self.delta_window:
            return 0
        return np.mean(self.delta_window)
