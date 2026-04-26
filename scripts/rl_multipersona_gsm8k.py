"""Entry point for the Multi-Persona Panel of Experts RL experiment.

Runs pure-RL-from-base on Qwen/Qwen3-30B-A3B-Base using the GSM8K
training set, shaped only by:
  (a) the prompt-prefill template that casts the generator as a
      Multi-Persona Panel of Experts (see envs/multipersona_gsm8k.py),
  (b) an outcome-only reward: correct numeric answer + <mutipersonaDebate>/
      <answer> tag enforcement (no process reward).

Usage:
    python scripts/rl_multipersona_gsm8k.py \\
        model_name=Qwen/Qwen3-30B-A3B-Base \\
        lora_rank=32 \\
        n_train=1024 n_test=128 \\
        group_size=8 groups_per_batch=16 \\
        learning_rate=5e-6 \\
        max_tokens=4096 \\
        max_steps=80
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make repo root importable so `envs.multipersona_gsm8k` resolves when the
# script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from envs.multipersona_gsm8k import MultipersonaGsm8kDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # --- model ---
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    lora_rank: int = 32
    renderer_name: str | None = "role_colon"  # metadata only; env builds its own prompt
    load_checkpoint_path: str | None = None  # pure RL from base by default

    # --- data ---
    seed: int = 20260423
    n_train: int | None = None      # None = use full GSM8K train split (~7.5k)
    n_test: int | None = 500        # subset of the 1319-example test split for eval

    # --- optimizer / batch ---
    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 5e-6
    temperature: float = 1.0
    max_tokens: int = 4096
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # --- reward shaping ---
    tag_coef: float = 0.2

    # --- schedule / IO ---
    max_steps: int | None = 80
    save_every: int = 10
    eval_every: int = 0   # disable in-training eval by default; eval with scripts/eval_gsm8k.py

    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    compute_post_kl: bool = False

    base_url: str | None = None


async def cli_main(cfg: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
    )

    run_name = (
        f"multipersona_gsm8k-{cfg.model_name.replace('/', '-')}-"
        f"{cfg.lora_rank}rank-lr{cfg.learning_rate}-g{cfg.group_size}-"
        f"b{cfg.groups_per_batch}-seed{cfg.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/rl/multipersona_gsm8k/{run_name}"

    dataset_builder = MultipersonaGsm8kDatasetBuilder(
        batch_size=cfg.groups_per_batch,
        model_name_for_tokenizer=cfg.model_name,
        group_size=cfg.group_size,
        seed=cfg.seed,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        tag_coef=cfg.tag_coef,
    )

    config = Config(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        log_path=log_path,
        base_url=cfg.base_url,
        load_checkpoint_path=cfg.load_checkpoint_path,
        compute_post_kl=cfg.compute_post_kl,
        kl_penalty_coef=cfg.kl_penalty_coef,
        num_substeps=cfg.num_substeps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cfg.max_steps_off_policy,
            groups_per_batch=cfg.groups_per_batch,
        )
        if cfg.max_steps_off_policy is not None
        else None,
        max_steps=cfg.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
