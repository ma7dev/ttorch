import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn.functional as F
from model import Net

class Module(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = Net()
        self.optim = kwargs['optim']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer, self.scheduler = self.configure_optimizers()

    def model_params(self):
        return self.model.parameters()

    def configure_optimizers(self):
        params = self.model_params()
        optimizer = None
        if self.optim.optimizer == 'sgd':
            optimizer = optim.SGD(
                params, lr=self.optim.lr, momentum=self.optim.momentum
            )
        elif self.optim.optimizer == 'adam':
            optimizer = optim.Adam(
                params, lr=self.optim.lr, betas=(0.9, 0.999), eps=1e-08
            )
        scheduler = None
        if self.optim.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.optim.step_size, gamma=self.optim.gamma
            )
        elif self.optim.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.optim.milestones, gamma=self.optim.gamma
            )
        elif self.optim.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.optim.gamma
            )
        return optimizer, scheduler
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def training_step(self, batch):
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        self.backward(loss)
        return z, y, loss
    
    def predicting_step(self, batch):
        x, y = batch
        z = self.forward(x)
        loss = self.criterion(z, y)
        return z, y, loss