import torch
from torch import nn
import numpy as np
import os
import pickle

from collections import namedtuple

from approximator import FunctionApproximator

Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])

def extract_trajectory_from_file(filepath):
    # expect file to contain s,a,r,s'
    # where each member is a list
    # we use pickle
    f = open(filepath, "rb")
    traj = pickle.load(f)
    return traj

def sample_trajectory(dirpath):
    files = os.listdir(dirpath)
    l = len(files)
    i = np.random.choice(l)
    return extract_trajectory_from_file(dirpath+"/"+files[i])

def fetch_all_trajectories(dirpath):
    files = os.listdir(dirpath)
    t = [ extract_trajectory_from_file(dirpath+"/"+f) for f in files]
    return t 

def printit(slist, traj, reward_model, cost_model):
    pred_reward = reward_model.cumsum(slist)
    pred_cost = cost_model.cumsum(slist)
    actual_reward = sum(map(lambda x : x[2], traj))
    actual_cost = sum(map(lambda x : x[3], traj))
    print("Traj: Predicted (r,c) = {}, {}; Actual (r,c) = {}, {}".format(pred_reward, pred_cost, actual_reward, actual_cost))


