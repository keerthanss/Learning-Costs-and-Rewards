import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import argparse

from collections import namedtuple

Transition = namedtuple('Transition', ['s', 'a', 'r', 'c', 'sprime', 'done'])

class FunctionApproximator(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        self.lr = 2.5e-3
        self.lambd = 0.001
        self.nopenaltyafter = 100 # Hardcoded. TODO
        self.count = 0
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)

    def forward(self, x, dropout=False):
        if dropout:
            x = F.relu(self.fc1(x))
            x = F.dropout(x)
            x = F.relu(self.fc2(x))
            x = F.dropout(x)
            x = torch.abs(self.fc3(x))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.abs(self.fc3(x))
        return x

    def cumsum(self, slist, dropout=False):
        with torch.no_grad():
            state_batch = torch.tensor(slist, dtype=torch.float32)
            return_batch = torch.sum(self.forward(state_batch, dropout))
        return return_batch


    def learn(self, slist1, slist2):
        # assuming slist1 and slist2 are from trajectories
        # t1 and t2 such that t1 < t2

        state_batch1 = torch.tensor(slist1, dtype=torch.float32)
        state_batch2 = torch.tensor(slist2, dtype=torch.float32)

        return_batch1 = torch.sum(self.forward(state_batch1, dropout=True))
        return_batch2 = torch.sum(self.forward(state_batch2, dropout=True))

        #loss = -torch.exp( return_batch2 - torch.log(torch.exp(return_batch1) + torch.exp(return_batch2)) )
        
        #l2 = torch.exp(return_batch2)
        #l1 = torch.exp(return_batch1)
        #loss = -l2 / (l2 + l1)
        #avg_exp_loss = l2 / (l2 + l1)

        loss = torch.log(1+torch.exp(return_batch1 - return_batch2)) + self.lambd*torch.square(return_batch1 + return_batch2)
        
        #if self.count < self.nopenaltyafter:
        #    loss = -1/(1 + torch.exp(return_batch1 - return_batch2)) + self.lambd * torch.square(return_batch1 + return_batch2)
        #else:
        #    loss = -1/(1 + torch.exp(return_batch1 - return_batch2))
        
        self.count += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Ensemble:

    def __init__(self, in_dim, n=5):
        self.fa_list = [FunctionApproximator(in_dim)]*n
        self.n = n

    def forward(self,x):
        result = 0
        for fa in self.fa_list:
            result += fa(x)
        result /= self.n
        return result
    
    def cumsum(self,x):
        result = np.average([fa.cumsum(x) for fa in self.fa_list])
        return result
    
    def learn(self, slist1, slist2):
        to_train = np.random.choice(self.n, 2, replace=False)
        loss = 0
        for i in to_train:
            loss += self.fa_list[i].learn(slist1, slist2)
        return loss / 2

    def save(self, epoch, savepath=".", savename="models"):
        torch.save({
            'epoch':epoch,
            'n':self.n,
            'fa_state_dict':[fa.state_dict() for fa in self.fa_list],
        }, savepath+"/{}_{}.pt".format(savename,epoch))
        return 

    
