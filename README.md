# Learning Costs and Rewards via Preference-based Learning [WiP]

This repository contains the code as well as the report detailing our investigation into learning costs and rewards simultaneously from a single set of expert demonstrations to aid in performing some task in the given environment. For further details, please read the report.

Data in the form of expert trajectories is provided Note that data/trajectories_training_set corresponds to expert demonstrations on OpenAI's Safety Gym's Goal-v1 environment using a Point robot. Further details can be found [here](https://openai.com/blog/safety-gym/). To generate the experts, we use the implementation provided [here](https://github.com/openai/safety-starter-agents). We slightly tweak the same implementation when training agents on our learned reward and cost functions. The code for the tweaking is not provided here.

Additionally, a gridworld environment called _drone pickup_ is implemented and used as a testbed as well. The expert trajectories for this can be found at drone_pickup_env/trajectories.zip. 
