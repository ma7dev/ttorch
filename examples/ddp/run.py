# credits:
# how to use DDP module with DDP sampler: https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
# how to setup a basic DDP example from scratch: https://pytorch.org/tutorials/intermediate/dist_tuto.html
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math

def get_dataset():
    world_size = dist.get_world_size()
    train_set = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    val_set = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    
    train_sampler = DistributedSampler(train_set,num_replicas=world_size)
    val_sampler = DistributedSampler(val_set,num_replicas=world_size)
    batch_size = int(128 / float(world_size))
    print(world_size, batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset=val_set,
        sampler=val_sampler,
        batch_size=batch_size
    )

    return train_loader, val_loader, batch_size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
def reduce_dict(input_dict, average=True):
    world_size = float(dist.get_world_size())
    names, values = [], []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values /= world_size
    reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
def train(model,train_loader,optimizer,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    train_num_batches = int(math.ceil(len(train_loader.dataset) / float(batch_size)))
    model.train()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = nn.CrossEntropyLoss().to(device)
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # average gradient as DDP doesn't do it correctly
        average_gradients(model)
        optimizer.step()
        loss_ = {'loss': torch.tensor(loss.item()).to(device)}
        train_loss += reduce_dict(loss_)['loss'].item()
        # cleanup
        # dist.barrier()
        # data, target, output = data.cpu(), target.cpu(), output.cpu()
    train_loss_val = train_loss / train_num_batches
    return train_loss_val
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.div_(batch_size))
        return res
def val(model, val_loader,batch_size):
    device = torch.device(f"cuda:{dist.get_rank()}")
    val_num_batches = int(math.ceil(len(val_loader.dataset) / float(batch_size)))
    model.eval()
    # let all processes sync up before starting with a new epoch of training
    # dist.barrier()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_ = {'loss': torch.tensor(loss.item()).to(device)}
            val_loss += reduce_dict(loss_)['loss'].item()
    val_loss_val = val_loss / val_num_batches
    return val_loss_val
def run(rank, world_size):
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234)
    train_loader, val_loader, batch_size = get_dataset()
    model = Net().to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # use if model contains batchnorm.
    model = DDP(model,device_ids=[rank],output_device=rank)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.5)
    history =  {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
    if rank == 0:
        history = {
            "rank": rank,
            "train_loss_val": [],
            "train_acc_val": [],
            "val_loss_val": [],
            "val_acc_val": []
        }
    for epoch in range(10):
        train_loss_val = train(model,train_loader,optimizer,batch_size)
        val_loss_val = val(model,val_loader,batch_size)
        if rank == 0:
            print(f'Rank {rank} epoch {epoch}: {train_loss_val:.2f}/{val_loss_val:.2f}')
            history['train_loss_val'].append(train_loss_val)
            history['val_loss_val'].append(val_loss_val)
    print(f'Rank {rank} finished training')
    print(history)
    cleanup(rank)  

def cleanup(rank):
    # dist.cleanup()  
    dist.destroy_process_group()
    print(f"Rank {rank} is done.")
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def init_process(
        rank, # rank of the process
        world_size, # number of workers
        fn, # function to be run
        # backend='gloo',# good for single node
        backend='gloo' # the best for CUDA
    ):
    # information used for rank 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    dist.barrier()
    setup_for_distributed(rank == 0)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()