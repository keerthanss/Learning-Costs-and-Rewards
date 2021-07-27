from gridworld import GridWorld, Action
import numpy as np
import pickle
import os
import shutil
from collections import namedtuple

Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])

def least_distance_action(o):
    # get perfect information
    agentx, agenty = o[0], o[1]
    deliveryx, deliveryy = o[2], o[3]
    carrying = o[4]
    battery_level = o[5]

    if carrying == 0:
        destinationx, destinationy = 0, 4
    else:
        destinationx, destinationy = deliveryx, deliveryy

    #calculate distance to charging station
    steps_to_charging = agentx + agenty
    steps_to_destination = np.abs(destinationx - agentx) + np.abs(destinationy - agenty)
    steps_to_charging_from_destination = destinationx + destinationy

    if battery_level < steps_to_destination + steps_to_charging_from_destination:#or battery_level == steps_to_charging:
        # don't have enough charge for round trip, head back to charge first
        destinationx, destinationy = 0,0

    movex, movey = np.sign(destinationx - agentx), np.sign(destinationy - agenty)
    action = None
    if movex != 0:
        action = Action.LEFT if movex == -1 else Action.RIGHT
    else:
        action = Action.DOWN if movey == -1 else Action.UP

    return action


def perfect_agent(num_episodes, threshold, dirpath):
    assert (threshold >= 30), "Battery level not sufficient to travel the world"
    env = GridWorld(threshold)
    max_ep_len = 200
    average_reward, average_cost = 0,0
    average_delivered = 0

    for ep in range(num_episodes):
        o = env.reset()
        done, carrying = False, False
        total_reward, total_cost = 0, 0
        traj = []
        steps = 0
        while not done and steps < max_ep_len:
            steps += 1
            action = least_distance_action(o)
            # perform the action
            oprime, r, done, info = env.step(action)

            data = Transition(o, action, r, info.get('cost', 0), oprime, done)
            traj.append(data)
            o = oprime

            total_reward += r
            total_cost += info['cost']
            '''
            print(action, destinationx, destinationy)
            env.render()
            print(r, info['cost'])
            '''
        f = open("{}/{}.pickle".format(dirpath, ep), "wb")
        pickle.dump(traj, f)
        f.close()
        traj = []
        
        average_reward += total_reward
        average_cost += total_cost
        average_delivered += env.delivered
        '''
        print("Episode complete")
        print(total_reward, total_cost)
        '''

    print("Performance with threshold {}".format(threshold))
    average_reward = average_reward / num_episodes
    average_cost = average_cost / num_episodes
    average_delivered = average_delivered / num_episodes
    print("Average reward = ", average_reward)
    print("Average cost = ", average_cost)
    print("Average delivered = ", average_delivered)
    return

if __name__=='__main__':
    if os.path.exists("trajectories"):
        shutil.rmtree("trajectories")
    os.mkdir("trajectories")

    for threshold in range(30,101,5):
        dirname = "trajectories/{}".format(threshold)
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
        perfect_agent(500, threshold, "trajectories/{}".format(threshold))

