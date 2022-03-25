import logging
from rich.logging import RichHandler
import wandb
import logging
import random
import string
from datetime import datetime
import os, copy
from omegaconf import OmegaConf

from logging import Filter
from logging.handlers import QueueListener
from multiprocessing import Queue
# credits:
# torch.distributed and logger: https://gist.github.com/scarecrow1123/967a97f553697743ae4ec7af36690da6
class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True
class Logger(object):
    def __init__(self, project, exp_name,save_dir,kwargs={},distributed=False,enable_wandb=False):
        # self.logger = logging.getLogger("rich")
        self.project = project
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.kwargs = kwargs
        self.enable_wandb = enable_wandb
        self.distributed = distributed
        # self.init()
        # self.log_kwargs()
        # Multiprocessing queue to which the workers should log their messages
        self.log_queue = Queue(-1)
        # self.logger = logging.getLogger(__name__)
        self.format = "%(message)s"
        self.datafmt = "[%X]"
        self.set_handlers()
        print(f"Logger initialized",distributed)
        if self.distributed:
            self.listener = QueueListener(self.log_queue, self.handlers[0], self.handlers[1], respect_handler_level=True)
            self.listener.start()
        else:
            logging.basicConfig(
                format=self.format, 
                datefmt=self.datafmt, 
                handlers=self.handlers
            )
    def init(self):
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        today = datetime.today().strftime("%Y-%m-%d")
        curr_time = datetime.today().strftime("%H-%M")
        self.exp_path = f'{self.save_dir}/{today}/{curr_time}-{self.exp_name}-{random_str}'
        self.create_log(today)
        if self.enable_wandb:
            wandb.init(
                project=self.project,
                config=self.kwargs
            )
    def create_log(self, today):
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
    def log_kwargs(self):
        conf = copy.deepcopy(self.kwargs)
        conf['exp_path'] = self.exp_path
        conf = OmegaConf.create(conf)
        with open(f'{self.exp_path}/kwargs.yaml', 'w') as fp: 
            OmegaConf.save(config=conf, f=fp)
    def watch(self,model,log_freq=100):
        if self.enable_wandb:
            wandb.watch(model, log_freq=log_freq)
    def set_handlers(self):
        # shell_handler = RichHandler()
        info_handler = logging.FileHandler("info.log",mode="w")
        debug_handler = logging.FileHandler("debug.log",mode="w")
        # shell_handler.setLevel(logging.DEBUG)
        info_handler.setLevel(logging.INFO)
        debug_handler.setLevel(logging.DEBUG)
        # file_handler.setLevel(logging.DEBUG)
        # fmt_shell = '%(message)s'
        # fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'   
        # shell_formatter = logging.Formatter(fmt_shell)
        # file_formatter = logging.Formatter(fmt_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  
        # shell_handler.setFormatter(shell_formatter)
        # file_handler.setFormatter(file_formatter)
        info_handler.setFormatter(formatter)
        debug_handler.setFormatter(formatter)
        self.handlers = [info_handler, debug_handler]  
    def info(self, msg): 
        print('info............')
        self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warning(self, msg): self.logger.warning(msg)
    def debug(self, msg): self.logger.debug(msg)
    def exception(self, msg): self.logger.exception(msg)
    def log(self, loss):
        if self.enable_wandb:
            wandb.log({"loss": loss})