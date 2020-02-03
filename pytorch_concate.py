import scipy.io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
#torch.cuda.manual_seed(manualSeed)
#torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _init_fn():
    np.random.seed(manualSeed)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('linear') != -1:
        #m.weight.data.fill_(0.01)
        #m.bias.data.fill_(0.01)
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

class ConcatenatedModel(nn.Module):
    def __init__(self):
        super(ConcatenatedModel, self).__init__()
        self.fc_spatial = nn.Linear(4, 5)
        #torch.nn.init.xavier_uniform(self.fc_spatial.weight)
        self.fc_spectral = nn.Linear(4, 5)
        self.fc1 = nn.Linear(20, 5)
        self.fc2 = nn.Linear(5, 1)
        
        
    def forward(self, x_spat, x_spec):
        fc_spat = F.relu(self.fc_spatial(x_spat))
        fc_spec = self.fc_spectral(x_spec)
        out = torch.cat((fc_spat, fc_spec), 1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
	
Model = ConcatenatedModel().double()
Model.apply(init_weight)
criterion = nn.L1Loss()
#criterion2 = nn.L1Loss(reduce = False)
optimizer = optim.Adadelta(Model.parameters(), 
                           lr=1.0, 
                           rho=0.95, 
                           eps=1e-06, 
                           weight_decay=0)

# Load data matrix 
with np.load('data_IP.npz', allow_pickle = True) as f:
    data_1, data_2, data_3 = f['data_spatial'], f['data_spectral'], f['data_label']
b_size = 10
x_spatial = torch.from_numpy(data_1).double()
x_spectral = torch.from_numpy(data_2).double()
labels = torch.from_numpy(data_3).double()
dataset = TensorDataset(x_spatial, x_spectral, labels)
loader = DataLoader(dataset, batch_size=b_size, 
                    num_workers=0, shuffle=False, 
                    worker_init_fn=_init_fn)
num_batches = len(loader)
b = 0

for batch_idx, (x1, x2, y) in enumerate(loader):
    i = (batch_idx+1)*b_size
    if i % (145*145) == 0:
        b = b+1
        print(b)
    optimizer.zero_grad()
    output = torch.squeeze(Model(x1, x2))
    loss = criterion(output, y)
    error = (y - torch.round(output))
    if batch_idx == 0:
        E = error
    elif batch_idx == num_batches-1:
        E = torch.cat((E, error),0)
        E = E.detach().numpy()
        scipy.io.savemat('pytorch_IP.mat', mdict={'E': E})
    else:
        E = torch.cat((E, error),0)
    loss.backward()
    optimizer.step()
