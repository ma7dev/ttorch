import imp
import torch
import torch.nn as nn
import torch.optim as optim
from rich import pretty
pretty.install()
import tqdm
import tqdm.rich
import logging
import random
import string
from datetime import datetime
import yaml, json
import os, sys, time, copy, pickle
from omegaconf import OmegaConf
import wandb
import tensorboardX
# logger for wandb
class Logger(object):
    def __init__(self, wandb=False,tb=False,**kwargs):
        self.exp_name = kwargs["exp_name"].replace(" ","_")
        self.trainer_kwargs = kwargs["trainer_kwargs"]
        self.save_dir = self.trainer_kwargs['save_dir'] if self.trainer_kwargs['save_dir'][-1] != '/' else self.trainer_kwargs['save_dir'][:-1]
        self.loggers = {}
        self.set_exp_path()
        if wandb: self.init_wandb()
        if tb: self.init_tensorboard()
    def set_exp_path(self):
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        today = datetime.today().strftime("%Y-%m-%d")
        curr_time = datetime.today().strftime("%H-%M")
        self.exp_path = f'{self.save_dir}/{today}/{curr_time}-{self.exp_name}-{random_str}'
        # create dir if not exists
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        if not os.path.exists(f'{self.save_dir}/{today}'): os.makedirs(f'{self.save_dir}/{today}')
        if not os.path.exists(self.exp_path): 
            os.makedirs(self.exp_path)
            os.makedirs(f"{self.exp_path}/ckpts")
            os.makedirs(f"{self.exp_path}/optims")
            os.makedirs(f"{self.exp_path}/figs")
            os.makedirs(f"{self.exp_path}/logs")
        else:
            raise Exception(f"Experiment path {self.exp_path} already exists")
        conf = copy.deepcopy(self.kwargs)
        conf['exp_path'] = self.exp_path
        conf = OmegaConf.create(conf)
        with open(f'{self.exp_path}/kwargs.yaml', 'w') as fp: 
            OmegaConf.save(config=conf, f=fp)
        self.paths = {
            'exp_path': self.exp_path,
            'exp_name': self.exp_name,
            'save_dir': self.save_dir,
            'kwargs': f'{self.exp_path}/kwargs.yaml',
            'ckpts': f'{self.exp_path}/ckpts',
            'optims': f'{self.exp_path}/optims',
            'figs': f'{self.exp_path}/figs',
        }
    def init_wandb(self):
        wandb.init(
            # project=self.exp_name,
            # name=self.exp_name,
            # dir=self.exp_path,
            # id=self.exp_name,
            # config=self.trainer_kwargs,
            # group=self.exp_name,
        )
        self.loggers['wand'] = wandb
    def init_tensorboard(self):
        self.loggers['tb'] = tensorboardX.SummaryWriter(log_dir=self.exp_path)
    def step_log(self, mode,step_counter,history):
        if 'wandb' in self.loggers.keys():
            wandb.log({
                f"{mode}_loss": history[mode]['loss'][-1],
            },step=step_counter[mode])
        if 'tb' in self.loggers.keys():
            self.loggers['tb'].add_scalar(f"{mode}_loss", history[mode]['loss'][-1], step_counter[mode])