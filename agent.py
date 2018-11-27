import numpy as np
from collections import namedtuple, deque
import random, copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent:
    " Interacts with and learns from the environment "
    
    
    def __init__(self, action_size, state_size, hidden_layers = [256, 64], random_seed = 123):
        """ Initialize attributes of Agent 
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.action_size = action_size
        self.states_size = state_size
        self.seed = random.seed(random_seed)
        
        self.local_actor = Actor(action_size, state_size, random_seed, hidden_layers)
        self.target_actor = Actor()
        self.local_critic = Critic()
        self.target_critic = Critic()
        
        
    
class ReplayBuffer:
    " Internal memory of the agent "
    
    
    

