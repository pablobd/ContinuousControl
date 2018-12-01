import numpy as np
from collections import namedtuple, deque
import random, copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.target_actor = Actor(action_size, state_size, random_seed, hidden_layers)
        self.local_critic = Critic(action_size, state_size, random_seed, hidden_layers)
        self.target_critic = Critic(action_size, state_size, random_seed, hidden_layers)
        
        self.experiencies = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
    def step(self, state, action, reward, next_state, done):
        """ Save experience in replay memory, and use random sample from buffer to learn        
        Params
        ======
            state (int): state of the environment
            action (int): action chosen by agent
            reward (int):  reward given by environment after doing this action
            next_state (int): state of the environment after doing this action
            done (int): flag indicating if episode has finished after doing this action
        """
        
        self.experiences.add(state, action, reward, next_state, done)
        
        if len(self.experiences) > BATCH_SIZE:
            batch = self.experiences.sample()
            self.learn(batch)
        
        
    def act(self, state):
        """ Given a state choose an action
        Params
        ======
            state (int): state of the environment        
        """
        
        self.local_actor
        
        
    
    def learn(self):
        return None
    
    def reset(self):
        return None
    
    def soft_update(self):
        return None
    
    
        
    
class ReplayBuffer:
    " Internal memory of the agent "
    
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
    
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done']).float().to(device)
        self.memory = deque()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        " Add a new experience to memory "
        
        self.experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        " Randomly sample a batch of experiences from the memory "
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.from_numpy([exp.state for exp in batch if exp is not None]).float().to(device)
        actions = torch.from_numpy([exp.action for exp in batch if exp is not None]).float().to(device)
        rewards = torch.from_numpy([exp.reward for exp in batch if exp is not None]).float().to(device)
        next_states = torch.from_numpy([exp.next_state for exp in batch if exp is not None]).float().to(device)
        dones = torch.from_numpy([exp.done for exp in batch if exp is not None]).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        " Return the current size of internal memory. Overwrites the inherited function len. "
        
        return len(self.memory)
        
    
    

