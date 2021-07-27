## README

This is a conceptually simple gridowlrd environment. The agent controls a drone and the task is to deliver goods from a fixed warehouse to randomized delivery locations. The drone has limited fuel, and therefore must return to the charging station to refuel. The reward is proportional to delivery latency, and the cost is a proxy for the fuel consumption. The battery size is the cost threshold.

Although simple conceptually, this environment proved too difficult for current state of the art constrained RL algorithms. We have thus provided a handcoded expert which has supplied a set of trajectories that can be used for inverse RL training. A modified version of the gridworld is also present that makes use of the learned reward function instead of the default one. Thus, existing RL libraries can be directly used on the environments with no modification.
