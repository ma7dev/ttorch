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

class Trainer(object):
    def __init__(self, **exp_kwargs):
        self.exp_kwargs = copy.deepcopy(exp_kwargs)
        self.trainer_kwargs = self.exp_kwargs['trainer_kwargs']
        self.seed = self.exp_kwargs['trainer_kwargs']['seed'] if 'seed' in self.exp_kwargs['trainer_kwargs'].keys() else random.randint(1, 10000)
        self.history = {'train': {'loss': [], 'correct': []}, 'val': {'loss': [], 'correct': []}}
        self.epoch_history = {'train': {'loss': [], 'acc': []}, 'val': {'loss': [], 'acc': []}}
        # self.set_exp_path()
        self.init()
    
    def set_exp_path(self):
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        today = datetime.today().strftime("%Y-%m-%d")
        curr_time = datetime.today().strftime("%H-%M")
        exp_name = self.exp_kwargs["exp_name"].replace(" ","_")
        save_dir = self.trainer_kwargs['save_dir'] if self.trainer_kwargs['save_dir'][-1] != '/' else self.trainer_kwargs['save_dir'][:-1]
        self.exp_path = f'{save_dir}/{today}/{curr_time}-{exp_name}-{random_str}'
        # create dir if not exists
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if not os.path.exists(f'{save_dir}/{today}'): os.makedirs(f'{save_dir}/{today}')
        if not os.path.exists(self.exp_path): 
            os.makedirs(self.exp_path)
            os.makedirs(f"{self.exp_path}/ckpts")
            os.makedirs(f"{self.exp_path}/optims")
            os.makedirs(f"{self.exp_path}/figs")
            os.makedirs(f"{self.exp_path}/logs")
        else:
            raise Exception(f"Experiment path {self.exp_path} already exists")
        conf = copy.deepcopy(self.exp_kwargs)
        conf['exp_path'] = self.exp_path
        conf = OmegaConf.create(conf)
        with open(f'{self.exp_path}/exp_kwargs.yaml', 'w') as fp: 
            OmegaConf.save(config=conf, f=fp)

    def init(self):
        # wandb.init(config=args)
        # wandb.init()
        self.logger = logging.getLogger(__name__); self.logger.setLevel(logging.INFO)
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def reset(self, mode, module):
        if mode == 'train':
            module.model.train()
            torch.set_grad_enabled(True)
            self.history['train']['loss'], self.history['train']['correct'] = [], []
        else:
            module.model.eval()
            torch.set_grad_enabled(False)
            self.history['val']['loss'], self.history['val']['correct'] = [], []

    def store(self, mode, model_output):
        output, target, loss = model_output        
        self.history[mode]['loss'].append(loss.item())
        self.history[mode]['correct'].append((output.max(1)[1] == target).sum().item())

    def logging(self, mode):
        print(f"{mode} loss: {sum(self.history[mode]['loss']) / len(self.history[mode]['loss'])}")
        print(f"{mode} acc: {sum(self.history[mode]['correct']) / len(self.history[mode]['correct'])}")

    def save(self, module, epoch):
        torch.save(module.model.state_dict(), f'{self.exp_path}/ckpts/{epoch:03d}_model.pth')
        torch.save(module.model.optimizer.state_dict(), f'{self.exp_path}/optims/{epoch:03d}_optimizer.pth')

    def update_history(self, mode):
        total = len(self.history[mode]['loss'])
        if mode == 'train':
            total *= self.exp_kwargs['dataset_kwargs']['batch_size']
        else:
            if 'val_batch_size' in self.exp_kwargs['dataset_kwargs'].keys():
                total *= self.exp_kwargs['dataset_kwargs']['val_batch_size']
            else:
                total *= self.exp_kwargs['dataset_kwargs']['batch_size']
        self.epoch_history[mode]['loss'].append(sum(self.history[mode]['loss']) / total)
        self.epoch_history[mode]['acc'].append(sum(self.history[mode]['correct']) / total)
    
    def update_loop(self, loop, postfix, iter_num=None):
        if iter_num: loop.update(iter_num)
        loop.set_postfix(postfix)
    
    def reset_loop(self, loop, postfix, rich=False):
        loop.refresh()
        loop.reset()
        loop.set_postfix(postfix)
    
    def step_loop(self, mode, module, dataset, step_loop, epoch_loop, epoch):
        data_loader = None
        if mode == 'train':
            self.reset(mode, module); step_loop.set_description(f"Epoch {epoch}:  Training") 
            data_loader = dataset.train_loader()
            for batch_idx, batch in tqdm.tqdm(enumerate(data_loader)):
                self.store(
                    mode, 
                    module.training_step(batch)
                )
                if batch_idx % self.trainer_kwargs['log_interval'] == 0: 
                    self.update_loop(
                        step_loop, 
                        {f'{mode}_loss': self.history[mode]['loss'][-1]}, 
                        iter_num=self.trainer_kwargs['log_interval']
                    )
        else:
            self.reset('val', module); step_loop.set_description(f"Epoch {epoch}:  Validating")
            data_loader = dataset.val_loader()
            with torch.no_grad():
                for batch_idx, batch in tqdm.tqdm(enumerate(data_loader)):
                    self.store(
                        mode, 
                        module.predicting_step(batch)
                    )
                    if batch_idx % self.trainer_kwargs['log_interval'] == 0: 
                        self.update_loop(
                            step_loop, 
                            {f'{mode}_loss': self.history[mode]['loss'][-1]}, 
                            iter_num=self.trainer_kwargs['log_interval']
                        )
        # step
        left_over = len(data_loader) % self.trainer_kwargs['log_interval']
        self.update_loop(
            step_loop, 
            {f'{mode}_loss': self.history[mode]['loss'][-1]}, 
            iter_num=left_over
        )
        self.update_history(mode)
        self.update_loop(
            epoch_loop, 
            {
                f'{mode}_loss': self.epoch_history[mode]['loss'][-1], 
                f'{mode}_acc': self.epoch_history[mode]['acc'][-1]
            }
        )

    def fit(self, module, dataset):
        # validate that the model works
        step_loop = tqdm.tqdm(
            range(2), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', file=sys.stdout
        )
        with torch.no_grad():
            val_loader = dataset.val_loader()
            for batch_idx, batch in enumerate(val_loader):
                step_loop.update(); step_loop.set_description(f"Validator {batch_idx}")
                self.store('val', module.predicting_step(batch))
                if batch_idx+1 == 2: break
        epoch_loop = tqdm.tqdm(
            range(1, self.trainer_kwargs['max_epochs'] + 1), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', file=sys.stdout
        )
        step_loop = tqdm.tqdm(
            range(len(dataset.train_loader())+len(dataset.val_loader())), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', file=sys.stdout
        )
        self.reset_loop(
            epoch_loop, 
            {
                'train_loss': 0, 'train_acc': 0, 
                'val_loss': 0, 'val_acc': 0
            }
        )
        self.reset_loop(
            step_loop, 
            {'train_loss': 0, 'val_loss': 0}
        )
        for epoch in epoch_loop:
            epoch_loop.update(); epoch_loop.set_description(f"Epoch {epoch}")
            self.reset_loop(
                step_loop, 
                {'train_loss': 0, 'val_loss': 0}
            )
            self.step_loop('train', module, dataset, step_loop, epoch_loop, epoch)
            self.step_loop('val', module, dataset, step_loop, epoch_loop, epoch)
            # save
            if ( 
                self.trainer_kwargs['save_interval'] 
                and epoch % self.trainer_kwargs['save_interval'] == 0
            ):
                self.save(module, epoch)
            # update scheduler
            module.scheduler.step()
            print('EEpoch\n',flush=True)
            print(
                f"Epoch {epoch}: \t",
                f"train/val_loss: {self.epoch_history['train']['loss'][-1]:.04f}/{self.epoch_history['val']['loss'][-1]:.04f}\t",
                f"train/val_acc: {self.epoch_history['train']['acc'][-1]:.04f}/{self.epoch_history['val']['acc'][-1]:.04f}\n\n"
            )
            # wandb.log({
            #     "train_loss": self.epoch_history['train']['loss'][-1],
            #     "val_loss": self.epoch_history['val']['loss'][-1],
            #     "train_acc": self.epoch_history['train']['acc'][-1],
            #     "val_acc": self.epoch_history['val']['acc'][-1],
            # })