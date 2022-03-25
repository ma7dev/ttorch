import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from module import Module
from dataset import Dataset
from trainer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from rich import pretty, traceback
pretty.install(); traceback.install()
import torch.multiprocessing as mp
def get_transforms(mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
@hydra.main(config_path="cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    os.environ['HYDRA_FULL_ERROR'] = '1'
    config = OmegaConf.to_yaml(cfg)
    print(config)
    mp.set_start_method("spawn")
    root_dir = cfg.project.root_dir
    distributed = len(cfg.exp.gpus)>1 if 'gpus' in cfg.exp.keys() else False
    # if distributed: torch.multiprocessing.set_start_method("spawn", force=True)
    dataset_kwargs = {
        'batch_size': cfg.exp.train.batch_size,
        'val_batch_size': cfg.exp.val.batch_size,
        'num_workers': cfg.exp.num_workers
    }
    dataset = Dataset(
        root_dir, 
        datasets.MNIST,
        get_transforms, 
        distributed=distributed,
        **dataset_kwargs
    )
    module_kwargs = {
        'optim': cfg.exp.train.optim
    }
    module = Module(**module_kwargs)
    trainer_kwargs = {
        'epochs': cfg.exp.epochs,
        'log_interval': cfg.exp.log_interval,
        # 'save_interval': cfg.exp.save_interval,
        'save_interval': False,
        # default values
        'seed': cfg.exp.seed if 'seed' in cfg.exp.keys() else 1,
        'resume': cfg.exp.resume if 'resume' in cfg.exp.keys() else False,
        'resume_path': cfg.exp.resume_path if 'resume_path' in cfg.exp.keys() else None,
        'gpus': cfg.exp.gpus if 'gpus' in cfg.exp.keys() else [0],
        # additional
        'ip': cfg.exp.ip if 'ip' in cfg.exp.keys() else '127.0.0.1',
        'port': cfg.exp.port if 'port' in cfg.exp.keys() else '29500',
    }
    exp_kwargs = {
        'project': cfg.project.name,
        'save_dir': cfg.project.save_dir if 'save_dir' in cfg.project.keys() else '.',
        'exp_name': cfg.exp.name, 
        'trainer_kwargs': trainer_kwargs, 
        'module_kwargs': module_kwargs, 
        'dataset_kwargs': dataset_kwargs
    }
    trainer = Trainer(module,dataset,distributed=distributed,**exp_kwargs)
    trainer.fit()
if __name__ == "__main__":
    main()