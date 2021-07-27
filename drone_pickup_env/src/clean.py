import utils
import numpy as np

from collections import namedtuple
Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])



dir1 = "../data/trajectories_training_set"
dir2 = "../data/cleaned"

l = 3
trajectories = {} # cost : list of all trajectories
for cost_threshold in range(1,l+1):
    trajectories[cost_threshold] = utils.fetch_all_trajectories(dir1+"/"+str(cost_threshold))

def total_cost(t):
    return sum(map(lambda x : x[3], t))

def total_reward(t):
    return sum(map(lambda x : x[2], t))

def allok(t1, t2):
    return total_cost(t1) < total_cost(t2) and total_reward(t1) < total_reward(t2)

lens = [len(trajectories[i]) for i in range(1,l+1)]

mixmashes = [(1,2), (1,3), (2,3)]
count_arr = np.ones((3,101), dtype=np.bool)
count = 0
for i,j in mixmashes:
    for ii, ti in enumerate(trajectories[i]):
        for jj, tj in enumerate(trajectories[j]):
            if not allok(ti, tj):
                count_arr[i-1][ii] = False
print(count_arr)



