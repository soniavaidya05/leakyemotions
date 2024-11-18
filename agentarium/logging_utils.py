import abc
import csv
import numpy as np
import pathlib
import torch
import wandb

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from IPython.display import clear_output


class Logger(abc.ABC):
    @abc.abstractmethod
    def log(self, data: Dict, step: int) -> None:
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def flush(self):
        pass


class MultiLogger(Logger):
    def __init__(self, loggers: Sequence[Logger]):
        self.loggers = loggers

    def log(self, data: Dict, step: int) -> None:
        for logger in self.loggers:
            logger.log(data, step)

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def close(self):
        for logger in self.loggers:
            logger.close()


@dataclass
class CSVLogger(Logger):
    path: str
    keys: List[str]
    delimiter: str = "\t"

    def __post_init__(self):
        logging_path = pathlib.Path(self.path)
        self.file = open(self.path, "wa")
        self.logger = csv.DictWriter(
            self.file, fieldnames=["step"] + self.keys, delimiter=self.delimiter
        )
        if not logging_path.exists():
            self.logger.writeheader()

        self.rows = []

    def log(self, data: Dict, step: int) -> None:
        self.rows.append({"step": step, **data})

    def flush(self):
        for row in self.rows:
            self.logger.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


@dataclass
class WandBLogger(Logger):
    project: str
    entity: str
    wandb_config: Optional[Dict] = None
    wandb_init_path: str = "wandb_init.txt"
    debug: bool = False

    def __post_init__(self):
        init_path = pathlib.Path(self.wandb_init_path)

        if init_path.exists():
            print("Trying to resume")
            resume_id = init_path.read_text()
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.wandb_config,
                resume=resume_id,
            )
        else:
            # if the run_id doesn't exist, then create a new run
            # and write the run id the file
            print("Starting new")
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                config=self.wandb_config,
            )
            init_path.write_text(str(run.id))
        self.rows = []

    def log(self, data: Dict, step: int) -> None:
        if self.debug:
            print("Logging skipped in debug")
        else:
            self.rows.append({"step": step, "data": data})

    def flush(self):
        for row in self.rows:
            wandb.log(row["data"], step=row["step"])
        self.rows = []

    def close(self):
        wandb.finish()

# --------------------------- #
# region: Game data storage   #
# --------------------------- #

class GameLogger:
    '''
    Container for storing game variables.
    '''
    def __init__(self, max_epochs):
        self.epochs = []
        self.turns = []
        self.losses = []
        self.rewards = []
        self.max_epochs = max_epochs

    def clear(self):
        '''
        Clear the game variables.
        '''
        del self.epochs[:]
        del self.turns[:]
        del self.losses[:]
        del self.rewards[:]
    
    def record_turn(
        self,
        epoch: int,
        turn: int,
        loss: float | torch.Tensor,
        reward: int | float | torch.Tensor
    ):
        '''
        Record a game turn.
        '''
        self.epochs.append(epoch)
        self.turns.append(turn)
        self.losses.append(np.round(loss, 2))
        self.rewards.append(reward)

    def pretty_print(
            self,
            *flags,
            **kwargs
        ) -> None:
        '''
        Take the results from a given epoch (epoch #, turn #, loss, and reward) 
        and return a formatted string that can be printed to the command line.

        If `jupyter-mode` is passed in as a flag, variables need to be passed 
        in with the `kwargs`.
        '''
        
        if 'jupyter-mode' in flags:
            assert all(key in kwargs.keys() for key in ('epoch', 'turn', 'reward')), 'Jupyter mode requires the current epoch, turn, and reward to be passed in as kwargs.'
            clear_output(wait = True)
            print(f'╔═════════════╦═══════════╦═════════════╦═════════════╗')
            print(f'║ Epoch: {str(kwargs["epoch"]).rjust(4)} ║ Turn: {str(kwargs["turn"]).rjust(3)} ║ Loss: {str("None").rjust(5)} ║ Reward: {str(kwargs["reward"]).rjust(3)} ║')
            print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝')
        else:
            if self.epochs[-1] == 0:
                print(f'╔═════════════╦═══════════╦═════════════╦═════════════╗')
            else:
                print(f'╠═════════════╬═══════════╬═════════════╬═════════════╣')
            if True:
                print(f'║ Epoch: {str(self.epochs[-1]).rjust(4)} ║ Turn: {str(self.turns[-1]).rjust(3)} ║ Loss: {str(self.losses[-1]).rjust(5)} ║ Reward: {str(self.rewards[-1]).rjust(3)} ║')
                print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝',end='\r')
            if self.epochs[-1] == self.max_epochs - 1:
                print(f'╚═════════════╩═══════════╩═════════════╩═════════════╝')

    def __repr__(self):
        return f'{self.__class__.__name__}(n_games={len(self.epochs)})'

# --------------------------- #
# endregion                   #
# --------------------------- #
    
