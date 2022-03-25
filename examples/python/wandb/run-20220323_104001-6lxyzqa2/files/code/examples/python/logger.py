# # import logging

# # logging.basicConfig(
# #     filename='app.log', 
# #     filemode='w', 
# #     format='%(name)s - %(levelname)s - %(message)s',
# #     datefmt='%d-%b-%y %H:%M:%S',
# #     level=logging.ERROR
# # )
# # logging.info('This will asdasdasdget logged to a file')
# # a = 5
# # b = 0

# # try:
# #   c = a / b
# # except Exception as e:
# #   logging.error("Exception occurred", exc_info=True)

# # # Create a custom logger
# # logger = logging.getLogger(__name__)

# # # Create handlers
# # c_handler = logging.StreamHandler()
# # f_handler = logging.FileHandler('file.log')
# # c_handler.setLevel(logging.WARNING)
# # f_handler.setLevel(logging.ERROR)

# # # Create formatters and add it to handlers
# # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # c_handler.setFormatter(c_format)
# # f_handler.setFormatter(f_format)

# # # Add handlers to the logger
# # logger.addHandler(c_handler)
# # logger.addHandler(f_handler)

# # logger.warning('This is a warning')
# # logger.error('This is an error')

import logging
from rich.logging import RichHandler
import wandb
import logging
import random
import string
from datetime import datetime
import os, sys, time, copy, pickle
from omegaconf import OmegaConf

class Logger(object):
    def __init__(self, exp_name,save_dir,kwargs={}):
        # self.logger = logging.getLogger("rich")
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.init()
        self.log_kwargs(kwargs)
        wandb.init()
        self.logger = logging.getLogger(__name__)
        self.format = "%(message)s"
        self.datafmt = "[%X]"
        self.set_handlers()
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
    def log_kwargs(self,kwargs):
        conf = copy.deepcopy(kwargs)
        conf['exp_path'] = self.exp_path
        conf = OmegaConf.create(conf)
        with open(f'{self.exp_path}/kwargs.yaml', 'w') as fp: 
            OmegaConf.save(config=conf, f=fp)
    def watch(self,model,log_freq=100):
        wandb.watch(model, log_freq=log_freq)
    def set_handlers(self):
        shell_handler = RichHandler()
        file_handler = logging.FileHandler("debug.log",mode="w")
        # shell_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        fmt_shell = '%(message)s'
        fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'   
        shell_formatter = logging.Formatter(fmt_shell)
        file_formatter = logging.Formatter(fmt_file)  
        shell_handler.setFormatter(shell_formatter)
        file_handler.setFormatter(file_formatter)
        self.handlers = [shell_handler, file_handler]  
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warning(self, msg): self.logger.warning(msg)
    def debug(self, msg): self.logger.debug(msg)
    def exception(self, msg): self.logger.exception(msg)
    def log(self, loss):
        wandb.log({"loss": loss})
    
# FORMAT = "%(message)s"
# logging.basicConfig(
#     level="NOTSET", 
#     format=FORMAT, 
#     datefmt="[%X]", 
#     handlers=[RichHandler()]
# )

# log = logging.getLogger("rich")
# log.info("Hello, World!")
# log.error("[bold red blink]Server is shutting down![/]", extra={"markup": True})
# log.error("123 will not be highlighted", extra={"highlighter": None})

# logging.basicConfig(
#     level="NOTSET",
#     format="%(message)s",
#     datefmt="[%X]",
#     handlers=[RichHandler(
#         rich_tracebacks=True,
#         tracebacks_suppress=[hydra] # supress hydra's traceback
#     )]
# )
# import logging

# from rich.logging import RichHandler

# logger = logging.getLogger(__name__)

# # the handler determines where the logs go: stdout/file
# shell_handler = RichHandler()
# file_handler = logging.FileHandler("debug.log")

# logger.setLevel(logging.DEBUG)
# shell_handler.setLevel(logging.DEBUG)
# file_handler.setLevel(logging.DEBUG)

# # the formatter determines what our logs will look like
# fmt_shell = '%(message)s'
# fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

# shell_formatter = logging.Formatter(fmt_shell)
# file_formatter = logging.Formatter(fmt_file)

# # here we hook everything together
# shell_handler.setFormatter(shell_formatter)
# file_handler.setFormatter(file_formatter)

# logger.addHandler(shell_handler)
# logger.addHandler(file_handler)

# log = logging.getLogger("rich")
exp_name = 'test'
save_dir = './logs'
log = Logger(exp_name,save_dir)
print('asdasdsads')

log.warning('asdasdasd')
log.error('asdasdasdasdasdas')
log.info('aasdsadasdasdasdasdasd asdas dasdas')
for i in range(10,100):
    log.log(i)
try:
    print(1 / 0)
except Exception:
    log.exception("unable print!")

# import logging

# from rich.console import Console
# from rich.logging import RichHandler

# console = Console(color_system=None)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(RichHandler(console=console,markup=True))

# string = "Foo [red on white]bar[/]"
# console.print(string) # Foo bar
# console.log(string) # [18:33:14] Foo bar                                         log.py:13
# # Should match the two above but doesn't
# logger.info(string) # [03/02/22 18:34:11] INFO     Foo [red on white]bar[/]      log.py:16