# Report

## Solution and Results

The algorithm used is DDPG described in this [paper](https://arxiv.org/abs/1509.02971). PyTorch is used to build the four neural networks. Specifically two Actor networks (which estimate best actions) and two Critic networks (which estimate discounted reward given a state and an actions).

Unfortunately, with this approach the agent learns very slowly, hence seems not suitable for this challenge (see the image below).


## Next Steps

As next step we target to use a parallel framework (several agents gaining experience from the environment in parallel) and applying the recently proposed algorithm D4PG. You can read about it in this [paper](https://openreview.net/forum?id=SyZipzbCb).
