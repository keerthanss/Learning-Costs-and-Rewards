from gym import spaces
from enum import IntEnum
from copy import deepcopy
from src.approximator import Ensemble
import numpy as np
import torch

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, p):
        return (self.x == p.x and self.y == p.y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def norm(self):
        return np.sqrt( self.x * self.x + self.y * self.y )

    def step(self, action):
        if action == Action.UP:
            self.y += 1
        elif action == Action.DOWN:
            self.y -= 1
        elif action == Action.LEFT:
            self.x -= 1
        elif action == Action.RIGHT:
            self.x += 1
        return


class GridWorld:

    def __init__(self, max_battery=-1):
        self.max_x = 10
        self.max_y = 5

        self.num_deliveries = 40
        self.start_pos = Point(1,0)
        self.charging_location = Point(0,0)
        self.warehouse_location = Point(0, self.max_y - 1)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2*(self.max_x + self.max_y),shape=(6,),dtype=np.int8)
        
        if max_battery == -1:
            self.max_battery = 2 * (self.max_x + self.max_y)
        else:
            self.max_battery = max_battery

        s = self.reset()

        self.reward_model = Ensemble(len(s),5)
        self.cost_model = Ensemble(len(s), 5)

        load_dir = "/home/keerthan/Projects/LCandR_FRESH/drone_pickup/data/runs/iter5k_bs4_seg100_tanh_256x256x1_E5/checkpoints/"
        self.reward_model.load(load_dir + "reward_4999.pt")
        self.cost_model.load(load_dir + "cost_4999.pt")
        return

    def _within_bounds(self, p):
        return ( p.x >= 0 and p.x < self.max_x and p.y >=0 and p.y < self.max_y)

    def _in_warehouse(self):
        return self.current_pos == self.warehouse_location

    def _in_charging_location(self):
        return self.current_pos == self.charging_location

    def _reached_target(self):
        return self.current_pos == self.delivery_location

    def _random_point(self):
        p = Point(-1, -1)
        while (not self._within_bounds(p)) or p == self.warehouse_location or p == self.charging_location or p == self.current_pos:
            x = np.random.randint(self.max_x)
            y = np.random.randint(self.max_y)
            p = Point(x,y)
        return p

    def _make_state(self):
        vec = self.delivery_location - self.current_pos
        state_list = [self.current_pos.x, self.current_pos.y, \
                        self.delivery_location.x, self.delivery_location.y, \
                        int(self.carrying_object), self.current_battery]
        return np.array(state_list, dtype=np.int8)

    def _incremental_reward(self):
        d1 = self.delivery_location - self.current_pos
        d2 = self.delivery_location - self.prev_pos
        return (d1.norm() - d2.norm()) 

    def step(self, action):
        self.prev_pos = deepcopy(self.current_pos)
        self.current_pos.step(action)
        done = False
        if not self._within_bounds(self.current_pos):
            self.current_pos = deepcopy(self.prev_pos)

        if self._in_charging_location():
            self.current_battery = self.max_battery
        else:
            self.current_battery -= 1 # fuel cost
            if self.current_battery == 0:
                done = True  # episode ends when there's no fuel

        if self._in_warehouse() and not self.carrying_object:
            # carry new object for delivery
            self.carrying_object = True

        if self.carrying_object and self._reached_target():
            self.delivered += 1
            if self.delivered == self.num_deliveries:
                done = True # episode ends if all deliveries are completed successfully
            self.carrying_object = False
            self.delivery_location = self._random_point()

        state = self._make_state()

        state_tensor = torch.tensor(state, dtype=torch.float32)
        reward = self.reward_model.forward(state_tensor).item()
        cost = self.cost_model.forward(state_tensor).item()
        info = {'cost':cost}
        return state, reward, done, info

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Render only supports human mode")
        for i in range(2*self.max_x+1):
            print("#", end="")
        print()

        print(".", end="")
        for i in range(self.max_x):
            print("_.", end="")
        print()

        for y in range(self.max_y - 1, -1, -1):
            print("|", end="")
            for x in range(self.max_x):
                p = Point(x,y)
                c = '_'
                if p == self.current_pos:
                    c = 'A'
                elif p == self.delivery_location:
                    c = 'D'
                elif p == self.charging_location:
                    c = 'C'
                elif p == self.warehouse_location:
                    c = 'W'
                print(c,end="|")
            print()

        for i in range(2*self.max_x + 1):
            print("#", end="")
        print()
        return

    def reset(self):
        self.delivered = 0
        self.current_battery = self.max_battery
        self.current_pos = self.start_pos
        self.delivery_location = self._random_point()
        self.prev_pos = None
        self.carrying_object = False
        return self._make_state()


def random_agent():
    env = GridWorld()
    for ep in range(5):
        o = env.reset()
        env.render()
        done = False

        while not done:
            action = np.random.randint(4)
            print(Action(action))
            o,r,done,info = env.step(action)
            env.render()
            print(r, info['cost'], o)
        print("Episode complete")

if __name__=="__main__":
    random_agent()
