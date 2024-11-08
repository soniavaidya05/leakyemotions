import abc
import csv
from dataclasses import dataclass
import pathlib
from typing import Dict, List, Optional, Sequence


import wandb


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