"""RL driver — PANEL arm on the olympiad union pool.

Continues training from the final panel-RL checkpoint (or a user-provided
checkpoint) on the PANEL-arm variance band produced by
scripts/build_per_arm_splits.py (default: data/olympiad_pool/panel_train.jsonl).

Pair with scripts/rl_thinking_olympiad.py — each arm trains on its own
variance band (largely disjoint on this pool), both arms score on the
same shared held-out eval (data/olympiad_pool/heldout_eval.jsonl).
Same hyperparameters across arms, different scaffold + training pool.

Usage:
    python scripts/rl_panel_olympiad.py                         # defaults
    python scripts/rl_panel_olympiad.py max_steps=100           # override
    python scripts/rl_panel_olympiad.py load_checkpoint_path=tinker://<your-ckpt>
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from envs.multipersona_olympiad import PanelOlympiadDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # --- model / init ---
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    lora_rank: int = 32
    renderer_name: str | None = "role_colon"
    # Default: final checkpoint from your panel-MATH RL run.
    # Override via PANEL_MATH_CHECKPOINT env var, or pass on CLI:
    #   bash scripts/rl_panel_olympiad.sh load_checkpoint_path=tinker://<your-session>:train:0/weights/final
    load_checkpoint_path: str = os.environ.get(
        "PANEL_MATH_CHECKPOINT",
        "tinker://<your-panel-math-session>:train:0/weights/final",
    )

    # --- data ---
    seed: int = 20260424
    train_path: str = "data/olympiad_pool/panel_train.jsonl"
    heldout_path: str = "data/olympiad_pool/heldout_eval.jsonl"
    virtual_train_len: int = 100_000

    # --- optimizer / batch ---
    group_size: int = 16                 # larger G for this experiment — variance-band sensitivity
    groups_per_batch: int = 8
    learning_rate: float = 5e-6
    temperature: float = 1.0
    max_tokens: int = 12288              # olympiad problems are longer
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # --- reward shaping ---
    tag_coef: float = 0.2

    # --- schedule ---
    max_steps: int | None = 200
    save_every: int = 10
    eval_every: int = 20

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
        f"panel_olympiad-{cfg.model_name.replace('/', '-')}-"
        f"{cfg.lora_rank}rank-lr{cfg.learning_rate}-g{cfg.group_size}-"
        f"b{cfg.groups_per_batch}-seed{cfg.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/rl/panel_olympiad/{run_name}"

    dataset_builder = PanelOlympiadDatasetBuilder(
        batch_size=cfg.groups_per_batch,
        model_name_for_tokenizer=cfg.model_name,
        group_size=cfg.group_size,
        seed=cfg.seed,
        tag_coef=cfg.tag_coef,
        train_path=cfg.train_path,
        heldout_path=cfg.heldout_path,
        virtual_train_len=cfg.virtual_train_len,
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
