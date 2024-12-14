# ------------------------ #
# region: Imports          #
import sys
import os
from datetime import datetime

# ------------------------ #
# region: path nonsense    #
# Determine appropriate paths for imports and storage
root = os.path.abspath('~/Documents/GitHub/agentarium') # Change the wd as needed.

# Make sure the transformers directory is in PYTHONPATH
if root not in sys.path:
    sys.path.insert(0, root)
# endregion                #
# ------------------------ #

from agentarium.primitives import Entity
from agentarium.config import load_config
from agentarium.models.iqn import iRainbowModel

from examples.cleanup.env import Cleanup
from examples.cleanup.agents import CleanupAgent

# endregion                #
# ------------------------ #

def run(cfg, **kwargs):
    # Set up the model and agents
    models = []
    agents = []
    for i in cfg.agent.agent.num:
        models.append(iRainbowModel(
            **cfg.model.iqn.parameters
        ))
        agents.append(CleanupAgent(
            cfg,
            model=models[i]
        ))
    # Set up the environment
    env = Cleanup(cfg, agents)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file", default="./configs/config.yaml")
    parser.add_argument("--load-weights", "-l", type=str, help="Path to pretrained model.", default="")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    cfg = load_config(args)
    run(
        cfg,
        load_weights=args.load_weights,
        save_weights=os.path.abspath(f'./models/checkpoints/{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl')
    )

