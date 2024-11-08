# --------------- #
# region: Imports #
# --------------- #

#Import base packages
import yaml
import os
import argparse

from typing import Optional, Sequence, Union

# --------------- #
# endregion       #
# --------------- #

# --------------------------- #
# region: config functions    #
# --------------------------- #

class Cfg:
    '''
    Configuration class for parsing the YAML configuration file.
    '''
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)

def init_log(cfg: Cfg) -> None:
    print(f'╔═════════════════════════════════════════════════════╗')
    print(f'║                                                     ║')
    print(f'║        Starting experiment: {str(cfg.experiment.name).ljust(16)}        ║')
    print(f'║        Epochs: {str(cfg.experiment.epochs).ljust(5)}  Max turns: {str(cfg.experiment.max_turns).ljust(3)}  Log: {"Yes" if cfg.log else "No "}      ║')
    print(f'║        Saving to: {str(cfg.save_dir).ljust(26)}        ║')
    print(f'║                                                     ║')
    print(f'╚═════════════════════════════════════════════════════╝')
    print(f'')

def parse_args(
        command_line: bool = True,
        args: Optional[Sequence[str]] = None
) -> argparse.Namespace:
    '''
    Helper function for preparsing the arguments.

    Parameters:
        args: (Optional) A sequence of command line-like arguments.
        By default, the function uses the default `--config` argument
        to get the file from `./config.yaml`. \n
        Unknown arguments (i.e., not `--config`) are ignored.
    
    Returns:
        The argparse.Namespace object containing the args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./config.yaml', help="path to config file")
    # By default, parse the arguments
    if command_line:
        args = parser.parse_args()
    # Otherwise, args can be passed in. Default value is specified for config.
    else:
        if args is None:
            args = argparse.Namespace(config='./config.yaml')
        else:
            args, _ = parser.parse_known_args(args)
    return args

def load_config(args: argparse.Namespace) -> Cfg:
    '''
    Load the parsed arguments into the Cfg class.

    Parameters:
        args: The argparse.Namespace object containing the args

    Returns:
        A Cfg class object with the configurations for the experiment
    '''
    if args.config is None or not os.path.isfile(args.config):
        raise ValueError("Config file not found, please make sure you've included a path to the config file.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = Cfg(config)
    
    return config
# --------------------------- #
# endregion: config functions #
# --------------------------- #