"""Stage-2 entry point for the matched <think> control.

Continues LoRA training from the stage-1 think-GSM8K checkpoint on the
MATH train split, with the same round-robin category-balanced sampling,
MATH-500 held out from the in-training eval, and same hyperparameters
as the panel MATH run.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from envs.think_math import ThinkMathDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # --- model / init ---
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    lora_rank: int = 32
    renderer_name: str | None = "role_colon"
    # Required at launch time — no default here since it depends on the
    # stage-1 run id (known only after stage-1 finishes).
    load_checkpoint_path: str | None = None

    # --- data ---
    seed: int = 20260423
    n_eval: int = 256
    n_train_per_subject: int | None = None
    virtual_train_len: int = 100_000

    # --- optimizer / batch (matched to panel MATH run) ---
    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 5e-6
    temperature: float = 1.0
    max_tokens: int = 8192
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # --- reward shaping ---
    tag_coef: float = 0.2

    # --- schedule ---
    max_steps: int | None = 128
    save_every: int = 10
    eval_every: int = 20

    log_path: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps_off_policy: int | None = None
    compute_post_kl: bool = False
    base_url: str | None = None


async def cli_main(cfg: CLIConfig):
    if not cfg.load_checkpoint_path:
        raise ValueError(
            "load_checkpoint_path is required for stage-2 (think MATH) — pass the "
            "state_path of the stage-1 think-GSM8K final checkpoint."
        )
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
    )

    run_name = (
        f"think_math-{cfg.model_name.replace('/', '-')}-"
        f"{cfg.lora_rank}rank-lr{cfg.learning_rate}-g{cfg.group_size}-"
        f"b{cfg.groups_per_batch}-seed{cfg.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/rl/think_math/{run_name}"

    dataset_builder = ThinkMathDatasetBuilder(
        batch_size=cfg.groups_per_batch,
        model_name_for_tokenizer=cfg.model_name,
        group_size=cfg.group_size,
        seed=cfg.seed,
        tag_coef=cfg.tag_coef,
        n_eval=cfg.n_eval,
        n_train_per_subject=cfg.n_train_per_subject,
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
