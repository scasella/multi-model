"""Entry point for continuation RL on MATH train from the GSM8K-trained checkpoint.

Loads weights from the batch-128 checkpoint of the multipersona GSM8K
run (e1a9b8bf-...) and continues LoRA training on the MATH train split.

Category-balanced round-robin sampling across the 7 MATH subjects:
every batch carries equal representation from each of algebra,
counting_and_probability, geometry, intermediate_algebra, number_theory,
prealgebra, precalculus.

Held-out in-training eval every `eval_every` steps on a fixed slice of
MATH-test with MATH-500 problems excluded, so MATH-500 stays clean as a
final post-training eval.

Defaults (match the GSM8K run unless noted):
    lora_rank=32, group_size=8, groups_per_batch=16, lr=5e-6,
    T=1.0, tag_coef=0.2, seed=20260423.
Changes:
    max_tokens=8192 (was 4096; MATH chains can be longer)
    max_steps=128   (matches GSM8K final run)
    eval_every=20   (in-training MATH-test holdout curve)

Usage:
    python scripts/rl_multipersona_math.py   # uses built-in defaults
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make repo root importable so `envs.multipersona_math` resolves when
# the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from envs.multipersona_math import MultipersonaMathDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # --- model / init ---
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    lora_rank: int = 32
    renderer_name: str | None = "role_colon"  # metadata only; env builds its own prompt
    # Training-state checkpoint from your multipersona_gsm8k final run (batch 128).
    # NB: use .../weights/final (training state), NOT .../sampler_weights/final.
    # Override with PANEL_GSM8K_CHECKPOINT env var, or pass on CLI:
    #   python scripts/rl_multipersona_math.py load_checkpoint_path=tinker://<your-session>:train:0/weights/final
    load_checkpoint_path: str = os.environ.get(
        "PANEL_GSM8K_CHECKPOINT",
        "tinker://<your-multipersona_gsm8k-session>:train:0/weights/final",
    )

    # --- data ---
    seed: int = 20260423
    n_eval: int = 256
    n_train_per_subject: int | None = None
    virtual_train_len: int = 100_000

    # --- optimizer / batch ---
    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 5e-6
    temperature: float = 1.0
    max_tokens: int = 8192
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # --- reward shaping ---
    tag_coef: float = 0.2

    # --- schedule / IO ---
    max_steps: int | None = 128
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
        f"multipersona_math-{cfg.model_name.replace('/', '-')}-"
        f"{cfg.lora_rank}rank-lr{cfg.learning_rate}-g{cfg.group_size}-"
        f"b{cfg.groups_per_batch}-seed{cfg.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/rl/multipersona_math/{run_name}"

    dataset_builder = MultipersonaMathDatasetBuilder(
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
