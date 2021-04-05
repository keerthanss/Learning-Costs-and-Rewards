import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FunctionApproximator(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        self.lr = 2.5e-3
        self.lambd = 0.001
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

    def learn(self, slist1, slist2, batch_size=1):
        # assuming slist1 and slist2 are from trajectories
        # t1 and t2 such that t1 < t2
        assert (batch_size >= 1), "Batch size passed is less than 1"

        state_batch1 = torch.tensor(slist1, dtype=torch.float32)
        state_batch2 = torch.tensor(slist2, dtype=torch.float32)

        axis = 0 if batch_size == 1 else 1
        return_batch1 = torch.sum(self.forward(state_batch1, dropout=True), axis=axis)
        return_batch2 = torch.sum(self.forward(state_batch2, dropout=True), axis=axis)

        loss = torch.log(1+torch.exp(return_batch1 - return_batch2)) + self.lambd*torch.square(return_batch1 + return_batch2)
        loss = loss.mean()
        
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
    
    def learn(self, slist1, slist2, batch_size=1):
        assert (batch_size >= 1), "Batch size passed is less than 1"
        to_train = np.random.choice(self.n, 2, replace=False)
        loss = 0
        for i in to_train:
            loss += self.fa_list[i].learn(slist1, slist2, batch_size)
        return loss / 2

    def save(self, epoch, savepath=".", savename="models"):
        torch.save({
            'epoch':epoch,
            'n':self.n,
            'fa_state_dict':[fa.state_dict() for fa in self.fa_list],
        }, savepath+"/{}_{}.pt".format(savename,epoch))
        return 

    def load(self, path):
        checkpoint = torch.load(path)
        fa_params = checkpoint['fa_state_dict']
        for i,fa in enumerate(self.fa_list):
            fa.load_state_dict(fa_params[i])
        return

    
