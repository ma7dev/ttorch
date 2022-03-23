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
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from torchvision import datasets, transforms
import random

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import math

def get_dataset(distributed):
    world_size = get_world_size()
    batch_size = int(128 / float(world_size))
    num_workers = 4
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
    train_sampler = RandomSampler(train_set)
    val_sampler = SequentialSampler(val_set)
    if distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)
    train_batch_sampler = BatchSampler(train_sampler, batch_size, drop_last=True)
    train_loader = DataLoader(
        dataset=train_set,
        sampler=train_batch_sampler,
        num_workers=num_workers,
        batch_size=batch_size
    )
    val_loader = DataLoader(
        dataset=val_set,
        sampler=val_sampler,
        num_workers=num_workers,
        batch_size=batch_size
    )

    return train_loader, val_loader, train_sampler, val_sampler, batch_size
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
def average_gradients(model):
    size = float(get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def is_main_process():
    return get_rank() == 0
def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
def run(rank, world_size, distributed):
    device = torch.device("cuda:{}".format(rank))
    group = dist.new_group([i for i in range(world_size)])
    epochs = 10
    torch.manual_seed(1234)
    train_loader, val_loader, train_sampler, val_sampler, batch_size = get_dataset(distributed)
    model = Net().to(device)
    model_without_ddp = model
    if distributed:
        model = DDP(model,device_ids=[rank],output_device=rank)
        model_without_ddp = model.module
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    train_num_batches = int(math.ceil(len(train_loader.dataset) / float(batch_size)))
    val_num_batches = int(math.ceil(len(val_loader.dataset) / float(batch_size)))
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
    for epoch in range(epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            # loss_mean = dist.reduce(loss, rank=0) / world_size
            # train_loss += loss_mean
            loss = loss.clone().detach()
            train_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        train_loss_val = train_loss / train_num_batches
        model.eval()
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        val_loss = 0.0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            # loss_mean = dist.reduce(loss, rank=0) / world_size
            # val_loss += loss_mean
            val_loss += loss.item()
        val_loss_val = val_loss / val_num_batches
        print(f'Rank {rank} epoch {epoch}: {train_loss_val:.2f}/{val_loss_val:.2f}')
        history['train_loss_val'].append(train_loss_val)
        # history['train_acc_val'].append(train_acc_val)
        history['val_loss_val'].append(val_loss_val)
        # history['val_acc_val'].append(val_acc_val)
    print(f'Rank {rank} finished training')
    print(history)
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
def init_distributed_mode(
        rank=-1, # rank of the process
        world_size=0, # number of workers
        fn=None, # function to be run
        # backend='gloo',# good for single node
        backend='nccl' # the best for CUDA
    ):
    ip = '127.0.0.1'
    port = '29500'
    dist_url = f'{ip}:{port}'
    if rank == -1 or world_size == 0:
        print('Not using distributed mode')
        return False

    dist.init_process_group(
        backend, 
        init_method=dist_url,
        rank=rank, 
        world_size=world_size
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True
def init_process(
    rank, # rank of the process
    world_size, # number of workers
    fn, # function to be run
    # backend='gloo',# good for single node
    backend='nccl' # the best for CUDA
    ):
    distributed = init_distributed_mode(rank, world_size, fn, backend)
    print(rank, world_size, distributed)
    fn(rank, world_size, distributed)


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