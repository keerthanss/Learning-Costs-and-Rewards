import numpy as np
import os
import pickle

from collections import namedtuple
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

def total_x(traj, i):
    return sum(map(lambda x : x[i], traj))

def total_reward(traj):
    return total_x(traj, 2)

def total_cost(traj):
    return total_x(traj, 3)

def _printit(traj_list, reward_model, cost_model):
    slist = list(map(lambda x : x[0], traj_list))
    pred_reward = reward_model.cumsum(slist)
    pred_cost = cost_model.cumsum(slist)
    actual_reward = total_reward(traj_list)
    actual_cost = total_cost(traj_list)
    print("Traj: Predicted (r,c) = {}, {}; Actual (r,c) = {}, {}".format(pred_reward, pred_cost, actual_reward, actual_cost))

def print_comparison(traj_dict, reward_model, cost_model):
    n = len(traj_dict.keys())
    # mini batch consists of trajectories from a single pair only
    random_threshold_selection = np.random.choice(n, 2, replace=False) + 1 #+1 due to folder naming starting from 1
    i, j = int(np.min(random_threshold_selection)), int(np.max(random_threshold_selection))

    _printit(traj_dict[i][np.random.choice(len(traj_dict[i]))], reward_model, cost_model)
    _printit(traj_dict[j][np.random.choice(len(traj_dict[j]))], reward_model, cost_model)
    return

def prune_preferences(traj_dict):
    n = len(traj_dict.keys())
    pruned_result = [] #set()
    all_ok = lambda t1, t2: total_reward(t1) <= total_reward(t2) and total_cost(t1) <= total_cost(t2)
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            for ii, ti in enumerate(traj_dict[i]):
                for jj, tj in enumerate(traj_dict[j]):
                    if all_ok(ti, tj):
                        pruned_result.append( ((i,ii),(j,jj)) )
    return pruned_result

def prepare_pruned_minibatch(traj_dict, traj_prefs, batch_size=64, segment_length=50):
    random_pairs = np.random.choice(len(traj_prefs), batch_size, replace = False)
    random_trajs1, random_trajs2 = [], []

    def random_trim(t):
        s = [ x[0] for x in t ]
        result = []
        if len(t) <= segment_length:
            result = s
            print("Warning: Trajectory length lesser than required")
        else:
            start = np.random.choice(len(t) - segment_length)
            result = s[start:start+segment_length]
        return result

    for k in random_pairs:
        (i,ii), (j,jj) = traj_prefs[k]
        random_trajs1.append( random_trim(traj_dict[i][ii]) )
        random_trajs2.append( random_trim(traj_dict[j][jj]) )

    i,j=1,2
    return i, j, random_trajs1, random_trajs2

def prepare_minibatch(traj_dict, batch_size=64, segment_length=50):
    n = len(traj_dict.keys())
    # mini batch consists of trajectories from a single pair only
    random_threshold_selection = np.random.choice(n, 2, replace=False) + 1 #+1 due to folder naming starting from 1
    i, j = int(np.min(random_threshold_selection)), int(np.max(random_threshold_selection))

    # select random trajectories within the selected thresholds
    random_trajs1 = np.random.choice(len(traj_dict[i]), batch_size, replace=False)
    random_trajs2 = np.random.choice(len(traj_dict[j]), batch_size, replace=False)

    def extract_trimmed(k, idx_list):
        trajs = traj_dict[k]
        result = []
        for idx in idx_list:
            t = trajs[idx]
            state_list = [ x[0] for x in t ]
            if len(t) <= segment_length:
                result.append(state_list)
                print("Warning: Trajectory length lesser than required")
            else:
                start = np.random.choice(len(t) - segment_length)
                result.append(state_list[start:start+segment_length])
        return result

    trimmed1 = extract_trimmed(i, random_trajs1)
    trimmed2 = extract_trimmed(j, random_trajs2)

    if batch_size == 1:
        # TODO: Bad design. Need to revisit later. For now
        # letting it be for backwards compatibility.
        trimmed1 = trimmed1[0]
        trimmed2 = trimmed2[0]

    return i,j, trimmed1, trimmed2

