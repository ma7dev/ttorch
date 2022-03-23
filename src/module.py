import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.nn.functional as F
from model import Net
import torch.distributed as dist
class Module(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = Net()
        self.optim = kwargs['optim']
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer, self.scheduler = self.configure_optimizers()

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
    def init_optimizer(self):
        self.optimizer, self.scheduler = self.configure_optimizers()
    def forward(self, x, distrubted=False):
        return self.model(x)
    def average_gradients(self):
        size = float(dist.get_world_size())
        for param in self.model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
    
    def backward(self, loss,distrubted=False):
        self.optimizer.zero_grad()
        loss.backward()
        if distrubted: self.average_gradients(self.model)
        self.optimizer.step()
        return loss
    
    def training_step(self, batch,distrubted=False):
        x, y = batch
        z = self.forward(x,distrubted)
        loss = self.criterion(z, y)
        self.backward(loss,distrubted)
        return z, y, loss
    
    def predicting_step(self, batch,distrubted=False):
        x, y = batch
        z = self.forward(x,distrubted)
        loss = self.criterion(z, y)
        return z, y, loss