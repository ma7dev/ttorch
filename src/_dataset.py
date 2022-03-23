import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class Dataset(object):
    def __init__(self, root, dataset, get_transforms, distributed=False, **kwargs):
        self.root = root
        self.kwargs = kwargs
        self.dataset = dataset
        self.get_transforms = get_transforms
        self.distributed = distributed
        if distributed: self.dist_sampler()
        self.reset()
        self.setup()
    
    def reset(self):
        if 'val_batch_size' not in self.kwargs.keys(): self.kwargs['val_batch_size'] = self.kwargs['batch_size']
        if 'num_workers' not in self.kwargs.keys(): self.kwargs['num_workers'] = 1
    
    def setup(self):
        print('==> Preparing data..')
        self.train_dataset = self.dataset(
            root=self.root, train=True, transform=self.get_transforms('train'), download=True
        )
        self.val_dataset = self.dataset(
            root=self.root, train=False, transform=self.get_transforms('val'), download=True
        )
    
    def dist_sampler(self):
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.val_sampler = DistributedSampler(self.val_dataset)

    def train_loader(self):
        if self.distributed:
            return DataLoader(
                self.train_dataset, 
                batch_size=int(self.kwargs['batch_size']/float(self.kwargs['world_size'])), 
                shuffle=True, 
                num_workers=self.kwargs['num_workers'],
                pin_memory=True, 
                sampler=self.train_sampler
            )
        return DataLoader(
            self.train_dataset, 
            batch_size=self.kwargs['batch_size'], 
            shuffle=True, 
            num_workers=self.kwargs['num_workers']
        )

    def val_loader(self):
        if self.distributed:
            return DataLoader(
                self.val_dataset, 
                batch_size=int(self.kwargs['val_batch_size']/float(self.kwargs['world_size'])), 
                shuffle=False, 
                num_workers=self.kwargs['num_workers'],
                pin_memory=True, 
                sampler=self.val_sampler
            )
        return DataLoader(
            self.val_dataset, 
            batch_size=self.kwargs['val_batch_size'], 
            shuffle=False, 
            num_workers=self.kwargs['num_workers']
        )