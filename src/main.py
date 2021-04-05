import torch
import numpy as np
import os
import argparse

from datetime import datetime
from collections import namedtuple

from approximator import FunctionApproximator, Ensemble
import utils

Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])

def train(dirpath, num_iters, save_freq=1000, save_path="."):
    # dirpath must contain numbered folders where the numbering
    # enforces the order 1 < 2 < 3 < 4 < ... so forth
    # each folder will contain any number of trajectories
    dirs = os.listdir(dirpath)
    l = len(dirs)

    # fetch state size by reading an arbitrary trajectory
    throwaway = utils.sample_trajectory(dirpath + "/1")
    state_size = len(throwaway[0][0])
    del throwaway

    reward = Ensemble(state_size) # FunctionApproximator(state_size)
    cost = Ensemble(state_size) #FunctionApproximator(state_size)

    trajectories = {} # cost : list of all trajectories
    for cost_threshold in range(1,l+1):
        trajectories[cost_threshold] = utils.fetch_all_trajectories(dirpath+"/"+str(cost_threshold))

    for epoch in range(num_iters):
        random_threshold_selection = np.random.choice(l, 2, replace=False) + 1
        i, j = int(np.min(random_threshold_selection)), int(np.max(random_threshold_selection))
        # i < j
        traj1 = trajectories[i][np.random.choice(len(trajectories[i]))]
        traj2 = trajectories[j][np.random.choice(len(trajectories[j]))]

        slist1 = list(map(lambda x : x[0], traj1))
        slist2 = list(map(lambda x : x[0], traj2))

        reward_loss = reward.learn(slist1, slist2)
        cost_loss = cost.learn(slist1, slist2)

        if epoch % save_freq == 0:
            reward.save(epoch, save_path, "reward")
            cost.save(epoch, save_path, "cost")
            print("Epoch {} : Reward loss = {}, Cost loss = {}".format(epoch, reward_loss, cost_loss))
            utils.printit(slist1, traj1, reward, cost)
            utils.printit(slist2, traj2, reward, cost)


def load(state_size, path):
    reward = FunctionApproximator(state_size)
    cost = FunctionApproximator(state_size)

    checkpoint = torch.load(path)
    reward.load_state_dict(checkpoint['reward_state_dict'])
    cost.load_state_dict(checkpoint['cost_state_dict'])

    return reward, cost



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Preference-based Inverse RL")
    parser.add_argument("--dir", help="Directory which contains the trajectories", default="../data/trajectories_training_set")
    parser.add_argument("--checkpoint_dir", help="Directory to store checkpoints", default="checkpoints")
    args = parser.parse_args()
    
    now = datetime.now()
    nowstr = now.strftime("%d-%m-%y_%H%M%S")
    main_dirname = "../data/runs/run_" + nowstr
    os.mkdir(main_dirname)

    chkptdir = main_dirname + "/" + args.checkpoint_dir
    os.mkdir(chkptdir)

    train(args.dir, 5000, 500, save_path=chkptdir)
