# --------------- #
# region: Imports #
# --------------- #
#Import base packages
import yaml
import os
import argparse

from typing import Optional, Sequence, Union

# Import gem packages
from examples.ft.models.iqn import iRainbowModel
from examples.ft.agents import Agent
from examples.ft.entities import Truck, Object
from examples.ft.utils import color_map
# --------------- #
# endregion       #
# --------------- #

# List of objects to generate

MODELS = {
    'iRainbowModel' : iRainbowModel
}

AGENTS = {
    'agent': Agent
}

ENTITIES = {
    'truck': Truck
}

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

# --------------------------- #
# region: object creators     #
# --------------------------- #
def create_models(cfg: Cfg) -> list[iRainbowModel]:
    '''
    Create a list of models used for the agents.

    Returns:
        A list of models of the specified type 
    '''
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])['type']]
        for _ in range(vars(vars(cfg.model)[model_name])['num']):
            model = MODEL_TYPE(**vars(cfg.model.iqn.parameters), device=cfg.model.iqn.device, seed = 1)
            model.name = model_name
            models.append(
                model
            )

    return models

def create_agents(
        cfg: Cfg, 
        models: list
    ) -> list[Agent]:
    '''
    Create a list of agents used for the task

    Parameters:
        models: A list of models that govern the agents' behaviour.

    Returns:
        A list of agents of the specified type
    '''
    agents = []
    model_num = 0
    colors = color_map(cfg.env.channels)
    for agent_type in vars(cfg.agent):
        AGENT_TYPE = AGENTS[agent_type]
        for _ in range(vars(vars(cfg.agent)[agent_type])['num']):

            # fetch for model in models
            agent_model_name = vars(vars(cfg.agent)[agent_type])['model']
            for model in models:
                has_model = False
                if model.name == agent_model_name:
                    agent_model = model
                    has_model = True
                    models.remove(model)
                
                if has_model:
                    break

            if not has_model:
                raise ValueError(f"Model {agent_model_name} not found, please make sure it is defined in the config file.")
            agents.append(AGENT_TYPE(
                color = colors['Agent'],
                model = agent_model,
                cfg = vars(cfg.agent)[agent_type]
            ))

        model_num += 1

    return agents

def create_entities(cfg: Cfg) -> list[Object]:
    '''
    Create a list of entities used for the task.

    Returns:
        A list of entities of the specified type
    '''
    entities = []
    colors = color_map(cfg.env.channels)
    for entity_type in vars(cfg.entity):
        ENTITY_TYPE = ENTITIES[entity_type]
        for entity in vars(cfg.entity)[entity_type]:
            color = colors[vars(entity)['cuisine']]
            entities.append(
                ENTITY_TYPE(
                    color=color,
                    cfg = entity
                )
            )
    return entities
# --------------------------- #
# endregion: object creators  #
# --------------------------- #