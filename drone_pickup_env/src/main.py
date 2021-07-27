import torch
import os
import argparse
import numpy as np

from datetime import datetime

from approximator import FunctionApproximator, Ensemble
import utils
from utils import Transition

def train(dirpath, num_iters, save_freq=500, save_path=".", batch_size=4, segment_length=100):
    # dirpath must contain numbered folders where the numbering
    # enforces the order 1 < 2 < 3 < 4 < ... so forth
    # each folder will contain any number of trajectories
    dirs = os.listdir(dirpath)
    l = len(dirs)
    intdirs = sorted(list(map(int, dirs)))

    # fetch state size by reading an arbitrary trajectory
    throwaway = utils.sample_trajectory(dirpath + "/" + str(intdirs[0]))
    state_size = len(throwaway[0][0])
    del throwaway

    reward = Ensemble(state_size) # FunctionApproximator(state_size)
    cost = Ensemble(state_size) #FunctionApproximator(state_size)

    trajectories = {} # cost : list of all trajectory locations
    for cost_threshold in intdirs:
        trajectories[cost_threshold] = utils.all_trajectory_locations(dirpath+"/"+str(cost_threshold))

    #proper_pairs = utils.prune_preferences(trajectories)

    for epoch in range(num_iters):
        #_, _, slist1, slist2 = utils.prepare_pruned_minibatch(trajectories, proper_pairs, batch_size=batch_size, segment_length=segment_length)
        threshold1, threshold2, slist1, slist2 = utils.prepare_minibatch(trajectories, batch_size, segment_length)
        reward_loss = reward.learn(slist1, slist2, batch_size=batch_size)
        cost_loss = cost.learn(slist1, slist2, batch_size=batch_size)#, bound1=threshold1, bound2=threshold2)

        if epoch % save_freq == save_freq - 1:
            reward.save(epoch, save_path, "reward")
            cost.save(epoch, save_path, "cost")
            print("Epoch {} : Reward loss = {}, Cost loss = {}".format(epoch, reward_loss, cost_loss))
            utils.print_comparison(trajectories, reward, cost)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Preference-based Inverse RL")
    parser.add_argument("--dir", help="Directory which contains the trajectories", default="../trajectories")
    parser.add_argument("--checkpoint_dir", help="Directory to store checkpoints", default="checkpoints")
    args = parser.parse_args()
    
    now = datetime.now()
    nowstr = now.strftime("%d-%m-%y_%H%M%S")
    main_dirname = "../data/runs/run_" + nowstr
    os.mkdir(main_dirname)

    chkptdir = main_dirname + "/" + args.checkpoint_dir
    os.mkdir(chkptdir)

    np.random.seed(10)
    train(args.dir, 5000, 100, save_path=chkptdir)
