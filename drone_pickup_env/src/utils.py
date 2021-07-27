import numpy as np
import os
import pickle
from gridworld import GridWorld
from collections import namedtuple
Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])

def extract_trajectory_from_file(filepath):
    # expect file to contain s,a,r,s'
    # where each member is a list
    # we use pickle
    f = open(filepath, "rb")
    traj = pickle.load(f)
    f.close()
    return traj

def sample_trajectory(dirpath):
    files = os.listdir(dirpath)
    l = len(files)
    i = np.random.choice(l)
    return extract_trajectory_from_file(dirpath+"/"+files[i])

def all_trajectory_locations(dirpath):
    #t = [ extract_trajectory_from_file(dirpath+"/"+f) for f in files]
    return [ (dirpath + "/" + file_) for file_ in os.listdir(dirpath)]

def total_x(traj, i):
    return sum(map(lambda x : x[i], traj))

def total_reward(traj):
    return sum(map(lambda x : x.r, traj))

def total_cost(traj):
    return sum(map(lambda x : x.c, traj))

def get_state_list(traj):
    return [ x.s for x in traj ]

def _printit(traj_loc, reward_model, cost_model):
    traj = extract_trajectory_from_file(traj_loc)
    slist = get_state_list(traj)
    pred_reward = reward_model.cumsum(slist)
    pred_cost = cost_model.cumsum(slist)
    actual_reward = total_reward(traj)
    actual_cost = total_cost(traj)
    print("Traj: Predicted (r,c) = {}, {}; Actual (r,c) = {}, {}".format(pred_reward, pred_cost, actual_reward, actual_cost))

def print_comparison(traj_dict, reward_model, cost_model):
    # mini batch consists of trajectories from a single pair only
    random_threshold_selection = np.random.choice(list(traj_dict.keys()), 2, replace=False)
    i, j = int(np.min(random_threshold_selection)), int(np.max(random_threshold_selection))

    _printit(traj_dict[i][np.random.choice(len(traj_dict[i]))], reward_model, cost_model)
    _printit(traj_dict[j][np.random.choice(len(traj_dict[j]))], reward_model, cost_model)
    return

def prune_preferences(traj_dict):
    keys = list(traj_dict.keys())
    pruned_result = [] #set()
    all_ok = lambda t1, t2: total_reward(t1) <= total_reward(t2) and total_cost(t1) <= total_cost(t2)
    for i in keys:
        for j in keys:
            for ii, ti in enumerate(traj_dict[i]):
                traji = extract_trajectory_from_file(ti)
                for jj, tj in enumerate(traj_dict[j]):
                    trajj = extract_trajectory_from_file(tj)
                    if all_ok(traji, trajj):
                        pruned_result.append( ((i,ii),(j,jj)) )
                    #elif all_ok(tj,ti):
                    #    pruned_result.append( ((j,jj), (i,ii)) )
    return pruned_result

def prepare_pruned_minibatch(traj_dict, traj_prefs, batch_size=64, segment_length=50):
    random_pairs = np.random.choice(len(traj_prefs), batch_size, replace = False)
    random_trajs1, random_trajs2 = [], []

    def random_trim(t):
        s = get_state_list(t)
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
    # mini batch consists of trajectories from a single pair only
    random_threshold_selection = np.random.choice(list(traj_dict.keys()), 2, replace=False)
    i, j = int(np.min(random_threshold_selection)), int(np.max(random_threshold_selection))

    # select random trajectories within the selected thresholds
    rpick = lambda k: np.random.choice(traj_dict[k], batch_size, replace=False)
    # extract the trajectories
    rextract = lambda traj_locs : list(map(extract_trajectory_from_file, traj_locs))

    # extract the trajectories
    random_trajs_i = rextract(rpick(i))
    random_trajs_j = rextract(rpick(j))

    # ensure lengths are greater than segment length
    # fetch least length
    min_len = lambda trajs: min([len(t) for t in trajs])
    least_length = min(map(min_len, [random_trajs_i, random_trajs_j]))
    if least_length < segment_length:
        print("Warning: Trajectory length lesser than required. Choosing smaller segment length.")
        segment_length = least_length
    
    # extract random segment
    start = np.random.choice(least_length - segment_length+1)
    # trim it
    trimit = lambda t: get_state_list(t)[start:start+segment_length]
    trimmed1 = list(map(trimit, random_trajs_i))
    trimmed2 = list(map(trimit, random_trajs_j))

    if batch_size == 1:
        # TODO: Bad design. Need to revisit later. For now
        # letting it be for backwards compatibility.
        trimmed1 = trimmed1[0]
        trimmed2 = trimmed2[0]

    return i,j, trimmed1, trimmed2

