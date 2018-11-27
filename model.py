import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    " Actor (Policy) Model - Neural net to decide what action the agent must take "
    
    def __init__(self, action_size, state_size, hidden_layers = [256, 64], seed = 123):
        """
        Initialize parameters and build model.
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        # initial layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # final layer
        self.fcfin = nn.Linear(hidden_layers[-1], action_size)
        
        self.reset_parameters()
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        
        # forward through each layer in `hidden_layers`, with ReLU activation
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        
        # forward final layer with tanh activation (-1, 1)
        return F.tanh(self.fcfin(x))
        
        