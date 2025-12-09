from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List

from omegaconf import OmegaConf, DictConfig

from sorrel.action.action_spec import ActionSpec
from sorrel.examples.leakyemotions.agents import LeakyEmotionsAgent
from sorrel.examples.leakyemotions.custom_observation_spec import (
    InteroceptiveObservationSpec,
    LeakyEmotionsObservationSpec,
    NoEmotionObservationSpec,
    OtherOnlyObservationSpec,
)
from sorrel.examples.leakyemotions.entities import EmptyEntity
from sorrel.examples.leakyemotions.env import LeakyEmotionsEnv
from sorrel.examples.leakyemotions.main import create_run_name, resolve_config_path
from sorrel.examples.leakyemotions.world import LeakyEmotionsWorld
from sorrel.models.pytorch.iqn import iRainbowModel
from sorrel.utils.logging import JupyterLogger
from sorrel.utils.visualization import ImageRenderer

ADULT_EPOCHS = 0
CHILD_EPOCHS = 100000

CHILD_COUNT_OPTIONS = [10]
ADULT_COUNT_OPTIONS = [0]
AGENT_MODES = ["child-only"] # ["child-only", "adult-child-both"]
BUSH_MODE = ["bush"]  # ["bush", "wolf", "both"]

EMOTION_CONDITIONS = ["full", "none"]
AGENT_VISION_RADIUS = [3]
SPAWN_PROBS = [0.0008, 0.001, 0.002, 0.003]
BUSH_LIFESPANS = [30, 35, 40, 50]

RUN_LABEL = "emotion_condition_ablations"
RUNS_ROOT = Path("runs_parent_child")
ENTITY_LIST = ["EmptyEntity", "Bush", "Wall", "Grass", "LeakyEmotionsAgent", "Wolf"]


def write_logger_csv(logger: JupyterLogger, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()
    logger.to_csv(csv_path)


def save_summary_checkpoint(records: List[Dict[str, float]], summary_path: Path) -> None:
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(summary_path, index=False)


def clone_config(cfg) -> DictConfig:
    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(config, DictConfig)
    return config

def build_observation_spec(condition: str, vision_radius: int):
    if condition == "full":
        spec = LeakyEmotionsObservationSpec(ENTITY_LIST, full_view=False, vision_radius=vision_radius)
    elif condition == "self":
        spec = InteroceptiveObservationSpec(ENTITY_LIST, full_view=False, vision_radius=vision_radius)
    elif condition == "other":
        spec = OtherOnlyObservationSpec(ENTITY_LIST, full_view=False, vision_radius=vision_radius)
    else:
        spec = NoEmotionObservationSpec(ENTITY_LIST, full_view=False, vision_radius=vision_radius)
    spec.override_input_size(np.zeros(spec.input_size, dtype=int).reshape(1, -1).shape)
    return spec


def create_untrained_agent(cfg) -> LeakyEmotionsAgent:
    observation_spec = build_observation_spec(cfg.model.emotion_condition, cfg.model.agent_vision_radius)
    action_spec = ActionSpec(["up", "down", "left", "right"])
    model = iRainbowModel(
        input_size=observation_spec.input_size,
        action_space=action_spec.n_actions,
        layer_size=250,
        epsilon=0.05,
        device="cpu",
        seed=torch.random.seed(),
        n_frames=5,
        n_step=3,
        sync_freq=200,
        model_update_freq=4,
        batch_size=64,
        memory_size=1024,
        LR=0.00025,
        TAU=0.001,
        GAMMA=0.99,
        n_quantiles=12,
    )
    return LeakyEmotionsAgent(observation_spec=observation_spec, action_spec=action_spec, model=model)


def freeze_agent_models(agent_list: List[LeakyEmotionsAgent]) -> None:
    for agent in agent_list:
        model = agent.model
        if hasattr(model, "eval"):
            model.eval()
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            for param in parameters():
                if hasattr(param, "requires_grad_"):
                    param.requires_grad_(False)


def run_child_training(
    env: LeakyEmotionsEnv,
    child_agents: List[LeakyEmotionsAgent],
    epochs: int,
    logger: JupyterLogger | None,
    animate: bool = False,
    output_dir: Path | None = None,
    csv_log_path: Path | None = None,
    csv_checkpoint_interval: int | None = None,
    writer: SummaryWriter | None = None,
    tensorboard_prefix: str = "child",
) -> None:
    assert child_agents, "child_agents list cannot be empty"
    max_turns = env.config.experiment.max_turns
    record_period = env.config.experiment.record_period
    epsilon_decay = getattr(env.config.model, "epsilon_decay", 0.0)
    checkpoint_interval = csv_checkpoint_interval or record_period or 1
    renderer = None
    if animate:
        if output_dir is None:
            output_dir = Path("animations")
        output_dir.mkdir(parents=True, exist_ok=True)
        renderer = ImageRenderer(
            experiment_name=env.world.__class__.__name__,
            record_period=record_period,
            num_turns=max_turns,
        )
    for epoch in range(epochs + 1):
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(epoch=epoch)
        animate_this_turn = animate and (epoch % record_period == 0)
        bunnies_left = sum(agent.alive for agent in env.bunnies)
        while env.turn < max_turns and bunnies_left > 0:
            if animate_this_turn and renderer is not None:
                renderer.add_image(env.world)
            env.take_turn()
            bunnies_left = sum(agent.alive for agent in env.bunnies)
        env.world.is_done = True
        if animate_this_turn and renderer is not None:
            renderer.save_gif(epoch, output_dir)
        total_loss = 0.0
        for agent in child_agents:
            loss = agent.model.train_step()
            total_loss += loss
            agent.model.epsilon_decay(epsilon_decay)
        if logger is not None:
            logger.record_turn(epoch, total_loss, env.world.total_reward, child_agents[0].model.epsilon)
            if csv_log_path is not None and (epoch % checkpoint_interval == 0 or epoch == epochs):
                write_logger_csv(logger, csv_log_path)
        if writer is not None:
            tag = tensorboard_prefix or "child"
            writer.add_scalar(f"{tag}/total_loss", total_loss, epoch)
            writer.add_scalar(f"{tag}/reward", env.world.total_reward, epoch)
            writer.add_scalar(f"{tag}/epsilon", child_agents[0].model.epsilon, epoch)
            writer.flush()


def configure_children(env: LeakyEmotionsEnv, cfg, child_count: int) -> List[LeakyEmotionsAgent]:
    child_agents = [create_untrained_agent(cfg) for _ in range(child_count)]
    for child in child_agents:
        child.model.train()
        env.agents.append(child)
        env.bunnies.append(child)
    cfg.world.agents = len(env.agents)
    return child_agents


def run_ablation_scenario(
    cfg,
    condition_label: str,
    mode_label: str,
    adult_count: int,
    child_count: int,
    base_dir: Path,
) -> Dict[str, str | float]:
    cfg.world.agents = adult_count
    cfg.experiment.run_name = create_run_name(cfg)
    world = LeakyEmotionsWorld(config=cfg, default_entity=EmptyEntity())
    env = LeakyEmotionsEnv(world, cfg)

    scenario_dir = base_dir / f"{mode_label}_adult{adult_count}_child{child_count}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = scenario_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    summaries: Dict[str, str | float] = {
        "condition": condition_label,
        "mode": mode_label,
        "adult_count": adult_count,
        "child_count": child_count,
    }

    try:
        if mode_label == "adult-child-both":
            adult_logger = JupyterLogger(max_epochs=ADULT_EPOCHS + 1)
            env.run_experiment(logger=adult_logger, animate=False)
            adult_rewards = np.array(adult_logger.rewards)
            np.save(scenario_dir / "adult_training_rewards.npy", adult_rewards)
            adult_log_path = scenario_dir / "adult_training_log.csv"
            write_logger_csv(adult_logger, adult_log_path)
            if adult_logger.losses:
                for epoch_idx, (loss, reward, epsilon) in enumerate(
                    zip(adult_logger.losses, adult_logger.rewards, adult_logger.epsilons)
                ):
                    writer.add_scalar("adult/total_loss", loss, epoch_idx)
                    writer.add_scalar("adult/reward", reward, epoch_idx)
                    writer.add_scalar("adult/epsilon", epsilon, epoch_idx)
                writer.flush()
            summaries["adult_final_reward"] = float(adult_rewards[-1])
            summaries["adult_best_reward"] = float(np.max(adult_rewards))
        else:
            summaries["adult_final_reward"] = np.nan
            summaries["adult_best_reward"] = np.nan

        if mode_label == "adult-child-both":
            adult_agents = list(env.agents)
            freeze_agent_models(adult_agents)
            child_agents = configure_children(env, cfg, child_count)
        elif mode_label == "child-only":
            env.agents = []
            env.bunnies = []
            cfg.world.agents = 0
            child_agents = configure_children(env, cfg, child_count)
        else:
            child_agents = []

        if child_agents:
            child_logger = JupyterLogger(max_epochs=CHILD_EPOCHS + 1)
            child_log_path = scenario_dir / "child_training_log.csv"
            run_child_training(
                env,
                child_agents,
                CHILD_EPOCHS,
                child_logger,
                animate=False,
                output_dir=None,
                csv_log_path=child_log_path,
                writer=writer,
                tensorboard_prefix="child",
            )
            guided_rewards = np.array(child_logger.rewards)
            np.save(scenario_dir / "child_training_rewards.npy", guided_rewards)
            write_logger_csv(child_logger, child_log_path)
            summaries["child_avg_reward"] = float(np.mean(guided_rewards))
        else:
            summaries["child_avg_reward"] = np.nan

        return summaries
    finally:
        writer.close()


if __name__ == "__main__":
    config_path = resolve_config_path("default.yaml")
    base_config = OmegaConf.load(config_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{RUN_LABEL}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "ablation_summary.csv"

    summaries = []
    for bush_lifespan in BUSH_LIFESPANS:
        for spawn_prob in SPAWN_PROBS:
            for avr in AGENT_VISION_RADIUS:
                for mode in BUSH_MODE:
                    for emotion_condition in EMOTION_CONDITIONS:
                        for adult_count in ADULT_COUNT_OPTIONS:
                            for child_count in CHILD_COUNT_OPTIONS:
                                for participation_mode in AGENT_MODES:
                                    if participation_mode == "child-only" and adult_count > 0:
                                        continue

                                    print("=== === === === === === === === === === === === === ===")
                                    print("===   Running ablation with the following parameters:  ===")
                                    print(f"===   Bush mode: {mode:<10}                                ===")
                                    print(f"===   Emotion condition: {emotion_condition:<8}                 ===")
                                    print(f"===   Agent vision radius: {avr:<2}                             ===")
                                    print(f"===   Bush spawn prob: {spawn_prob:<6}                          ===")
                                    print(f"===   Bush lifespan: {bush_lifespan:<3}                            ===")
                                    print(f"===   Adults: {adult_count:<2} | Children: {child_count:<2}               ===")
                                    print(f"===   Participation mode: {participation_mode:<16} ===")
                                    print("=== === === === === === === === === === === === === ===")

                                    cfg = clone_config(base_config)
                                    cfg.world.spawn_prob = spawn_prob
                                    cfg.world.bush_lifespan = bush_lifespan
                                    cfg.model.agent_vision_radius = avr
                                    cfg.model.emotion_condition = emotion_condition
                                    cfg.experiment.mode = mode
                                    if mode == "bush":
                                        cfg.world.wolves = 0
                                    elif mode == "wolf":
                                        cfg.world.spawn_prob = 0
                                        assert cfg.world.wolves > 0, "Must have nonzero number of wolves in wolf mode."

                                    condition_dir = run_dir / emotion_condition
                                    condition_dir.mkdir(parents=True, exist_ok=True)

                                    summary = run_ablation_scenario(
                                        cfg=cfg,
                                        condition_label=emotion_condition,
                                        mode_label=participation_mode,
                                        adult_count=adult_count,
                                        child_count=child_count,
                                        base_dir=condition_dir,
                                    )
                                    summary["spawn_prob"] = spawn_prob
                                    summary["vision_radius"] = avr
                                    summary["bush_mode"] = mode
                                    summary["bush_lifespan"] = bush_lifespan
                                    summaries.append(summary)
                                    save_summary_checkpoint(summaries, summary_path)

    if summaries:
        save_summary_checkpoint(summaries, summary_path)
        print(f"Ablation run complete. Summary saved to {summary_path}")
    else:
        print("No ablation combinations were executed.")
