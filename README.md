# ContinuousControl DDPG

## Description
Train a DRL agent control a double-jointed arm, to reach target locations. Watch this [YouTube video](https://www.youtube.com/watch?v=ZVIxt2rt1_4) to see how some researchers were able to train a similar task on a real robot! The accompanying research paper can be found [here](https://arxiv.org/pdf/1803.07067.pdf).

We implement the [DDPG](https://arxiv.org/abs/1509.02971) algorithm to train the agent. This algorithm turns out to be too slow, as the experience is not in parallel and the agent has difficulties to learn.

#### Rewards
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

#### States
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Solving the environment

**One agent.** The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.


## Requirements
First of all, you need python 3 and conda. We suggest to use the [Anaconda distribution](https://www.anaconda.com/download/#linux), although other options are available. Follow the instructions at the github [repository dlrn](https://github.com/udacity/deep-reinforcement-learning) to create an environment for the project, install dependencies and create a kernel. You will not need to install Unity, because Udacity provides two separate versions of the Unity environment:

* The first version contains a single agent.

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

For game developers, these trained agents can be used for multiple purposes, including controlling [NPC](https://en.wikipedia.org/wiki/Non-player_character) behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. 

#### One (1) Agent

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


#### Your own Unity environment

If you are interested in building your own Unity environments after completing the project, you can follow the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md), which walk you through all of the details of building an environment from a Unity scene. 

For game developers, these trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.


## How it works
Open the file  jupyter notebook Continuous_Control.ipynb and execute it reading carefully the instructions. Notice that when creating an environemnt for the game we forced the option no_graphics=True. You can change it to False to see a graphic representation of the game.




