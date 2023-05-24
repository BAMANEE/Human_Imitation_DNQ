"""
Various models for the agent to use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)        


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QnetworkImage(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QnetworkImage, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(4096, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, action_size)
                                        )

    def forward(self, state):
        print(state.shape)
        x = state.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

class QnetworkImageResnet(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QnetworkImageResnet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.resnet = timm.create_model('resnet18', num_classes=action_size)

    def forward(self, state):
        try:
            x = state.permute(0, 3, 1, 2)
        except:
            print(state.shape)
            print(state)
            x = state.permute(0, 3, 1, 2)
        x = self.resnet(x)
        return x
      

class ImitationNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ImitationNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        #return F.softmax(self.fc3(x), dim=-1)
        return self.fc3(x)

class ImitationNetworkImage(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(ImitationNetworkImage, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.resnet = timm.create_model('resnet18', num_classes=action_size)

        self.cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                    nn.ReLU(True),
                                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                nn.ReLU(True)
                                    )
        self.classifier = nn.Sequential(nn.Linear(4096, 512),
                                    nn.ReLU(True),
                                    nn.Linear(512, action_size)
                                    )
                            
                

    def forward(self, state):
        x = state.permute(0, 3, 1, 2)
        x = self.resnet(x)
        return F.softmax(x, dim=-1)
