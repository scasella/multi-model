"""Evaluate a multipersona-gsm8k checkpoint on the GSM8K test split.

Loads a trained sampler checkpoint (path of the form
``tinker://<session>:train:0/sampler_weights/<name>``) and runs rollouts
against the first ``n_test`` rows of the GSM8K test split with the same
prompt template used during training.

Scoring mirrors what the training loop logged:
  - ``correct``   : numeric match against the GSM8K gold answer
  - ``tag_valid`` : well-formed <mutipersonaDebate>...</mutipersonaDebate>
                    <answer>...</answer> pair
  - ``reward``    : correct + tag_coef * (tag_valid - 1.0)

One sample per problem at the configured temperature. This is directly
comparable to the training-time ``env/all/*`` metrics when temperature
matches the training temperature (default 1.0).

Usage:
    python scripts/eval_gsm8k.py \\
        sampler_path=tinker://<your-session>:train:0/sampler_weights/000080 \\
        tag=step80 n_test=500 temperature=1.0

Outputs:
    reports/eval_gsm8k/<tag>/summary.json
    reports/eval_gsm8k/<tag>/rollouts.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chz
import tinker

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from envs.multipersona_gsm8k import (  # noqa: E402
    MultipersonaGsm8kDataset,
    check_correct,
    check_tag_structure,
    extract_answer_text,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_gsm8k")


@chz.chz
class EvalConfig:
    # Checkpoint. Use the SAMPLER path (not the training-state path).
    sampler_path: str
    # Label for output directory (e.g. "step80", "step70", "base").
    tag: str
    # Dataset / generation.
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    n_test: int = 500
    batch_size: int = 16
    max_tokens: int = 4096
    temperature: float = 1.0
    tag_coef: float = 0.2
    seed: int = 20260423
    # Output.
    out_root: str = str(ROOT / "reports" / "eval_gsm8k")


async def _eval(cfg: EvalConfig) -> None:
    out_dir = Path(cfg.out_root) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    summary_path = out_dir / "summary.json"

    logger.info("sampler_path=%s", cfg.sampler_path)
    logger.info("tag=%s  n_test=%d  temperature=%g", cfg.tag, cfg.n_test, cfg.temperature)
    logger.info("out_dir=%s", out_dir)

    # --- Build dataset (split='test' forces group_size=1 inside the dataset).
    tokenizer = get_tokenizer(cfg.model_name)
    test_ds = MultipersonaGsm8kDataset(
        batch_size=cfg.batch_size,
        group_size=1,
        tokenizer=tokenizer,
        split="test",
        seed=cfg.seed,
        n_examples=cfg.n_test,
        tag_coef=cfg.tag_coef,
    )
    n_batches = len(test_ds)
    total_examples = min(cfg.n_test, 1319)
    logger.info("test_ds: %d examples across %d batches of size %d",
                total_examples, n_batches, cfg.batch_size)

    # --- Build sampling client from the checkpoint.
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=cfg.sampler_path)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    # --- Roll out.
    n_correct = 0
    n_tag = 0
    n_done = 0
    sum_reward = 0.0
    t_start = time.time()

    with open(rollouts_path, "w") as fh_rollouts:
        for batch_i in range(n_batches):
            group_builders = test_ds.get_batch(batch_i)
            logger.info("batch %d/%d  (problems %d..%d)",
                        batch_i + 1, n_batches,
                        batch_i * cfg.batch_size,
                        batch_i * cfg.batch_size + len(group_builders) - 1)

            # Run all groups in this batch concurrently.
            groups = await asyncio.gather(
                *[do_group_rollout(gb, policy) for gb in group_builders]
            )

            for gb, group in zip(group_builders, groups, strict=True):
                # group_size=1 → single trajectory
                trajectory = group.trajectories_G[0]
                # Pull the (sole) env back from the builder to know gold+problem.
                env = (await gb.make_envs())[0]
                # Trajectory.transitions[i].ac is a TokensWithLogprobs with .tokens.
                tokens: list[int] = []
                for tr in trajectory.transitions:
                    tokens.extend(tr.ac.tokens)
                text = tokenizer.decode(tokens)
                # If stop_reason was 'stop', env already accepted without </answer>;
                # here we grade using the text as-is with a tolerant closer.
                text_for_grading = text
                if "</answer>" not in text_for_grading and "<answer>" in text_for_grading:
                    text_for_grading = text + "</answer>"

                tag_valid = check_tag_structure(text_for_grading)
                correct = check_correct(text_for_grading, env.gold)
                reward = float(correct) + cfg.tag_coef * (float(tag_valid) - 1.0)
                ans = extract_answer_text(text_for_grading)

                n_tag += int(tag_valid)
                n_correct += int(correct)
                sum_reward += reward
                n_done += 1

                fh_rollouts.write(
                    json.dumps({
                        "idx": n_done - 1,
                        "problem": env.problem,
                        "gold": env.gold,
                        "extracted_answer": ans,
                        "tag_valid": bool(tag_valid),
                        "correct": bool(correct),
                        "reward": reward,
                        "completion_chars": len(text),
                        "completion_tokens": len(tokens),
                        "completion": text,
                    }) + "\n"
                )

            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-6)
            logger.info(
                "  progress %d/%d  correct=%.4f  tag_valid=%.4f  reward=%.4f  (%.1f ex/s)",
                n_done, total_examples,
                n_correct / n_done, n_tag / n_done, sum_reward / n_done, rate,
            )

    # --- Summary.
    total = n_done
    summary = {
        "tag": cfg.tag,
        "sampler_path": cfg.sampler_path,
        "model_name": cfg.model_name,
        "n_test_requested": cfg.n_test,
        "n_evaluated": total,
        "batch_size": cfg.batch_size,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "tag_coef": cfg.tag_coef,
        "seed": cfg.seed,
        "metrics": {
            "correct": n_correct / total,
            "tag_valid": n_tag / total,
            "reward_total": sum_reward / total,
        },
        "counts": {
            "n_correct": n_correct,
            "n_tag_valid": n_tag,
            "n_total": total,
        },
        "elapsed_sec": time.time() - t_start,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("=== FINAL for tag=%s ===", cfg.tag)
    logger.info("correct=%.4f  tag_valid=%.4f  reward=%.4f  (n=%d)",
                summary["metrics"]["correct"], summary["metrics"]["tag_valid"],
                summary["metrics"]["reward_total"], total)
    logger.info("wrote %s", summary_path)
    logger.info("wrote %s", rollouts_path)


if __name__ == "__main__":
    cfg = chz.entrypoint(EvalConfig)
    asyncio.run(_eval(cfg))
