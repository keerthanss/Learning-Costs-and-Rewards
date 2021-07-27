## README

The experiments have been conducted with openAI Safe RL gym environments as well as with a custom gridworld environment. The neural network models used are the same in both.

To work with the Safe RL environments, we made use of safety-starter-agents repository. For training with the learned reward function, safety-starter-agents/safe\_rl/pg/run\_agent.py needs to be modified slightly. This is fairly straightforward, check under "Run main environment interaction loop" section.

To train new models with safe RL algorithms, use safety-starter-agents/scripts/experiment.py. To obtain trajectories out of learned models, use test\_policy.py.
