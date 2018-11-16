import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ppoAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_layers = [64, 64], seed = 0):#drop_p = 0.8, 
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        # envrionmend dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # policy
        self.policy = Policy(state_size, action_size, hidden_layers, seed).to(device) 
        self.optimizer = optim.Adam(self.local_ppoNet.parameters(), lr = LR)
        
        # collection of trajectories - list of tuples: prob, state, action, reward
        self.experience = namedtuple("Trajectories", field_names = ["prob", "state", "action", "reward"])
        self.trajectories = deque()

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def collect_trajectories(envs, policy, tmax=100):
        """ collect trajectories with a given policy  """
        # number of parallel instances
        self.trajectories.clear()
        n=len(envs.ps)

        #initialize returning lists and start the game!
        state_list=[]
        reward_list=[]
        prob_list=[]
        action_list=[]

        envs.reset()
    
        # start all parallel agents
        envs.step([1]*n)
    
        # perform nrand random steps
        for _ in range(nrand):
            fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
            fr2, re2, _, _ = envs.step([0]*n)
    
        for t in range(tmax):

            # prepare the input
            # preprocess_batch properly converts two frames into 
            # shape (n, 2, 80, 80), the proper input for the policy
            # this is required when building CNN with pytorch
            batch_input = preprocess_batch([fr1,fr2])
        
            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
            action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
            probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        
            # advance the game (0=no action)
            # we take one action and skip game forward
            fr1, re1, is_done, _ = envs.step(action)
            fr2, re2, is_done, _ = envs.step([0]*n)

            reward = re1 + re2
        
            # store the result
            state_list.append(batch_input)
            reward_list.append(reward)
            prob_list.append(probs)
            action_list.append(action)
        
            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if is_done.any():
                break

        self.trajectoris = 
        # return pi_theta, states, actions, rewards, probability
        return prob_list, state_list, \
            action_list, reward_list

    def learn(deterministic = False):
        """ choose an action deterministic or stochastic """
        
        
        
    def surrogate(policy, old_probs, states, actions, rewards,
                  discount = 0.995, beta=0.01):        
        """ returns sum of log-prob divided by T same thing as -policy_loss """ 
        
        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

        ratio = new_probs/old_probs

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        return torch.mean(ratio*rewards + beta*entropy)
    

    
class Policy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers = [64, 64], seed = 0):
        """Initialize parameters and build model.
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
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
                
        self.olinear = nn.Linear(hidden_layers[-1], action_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        """Build a network that maps state -> action values."""
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        
        return self.sig(self.olinear(x))
    