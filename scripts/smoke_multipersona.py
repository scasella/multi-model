"""Smoke test for envs.multipersona_gsm8k.

Verifies:
  - tag_structure check accepts canonical and permissive variants, rejects malformed.
  - answer extraction handles closed/unclosed <answer> blocks and \\boxed{} wrappers.
  - GSM8K gold extraction returns the '####'-trailing value.
  - correctness grading matches int, float, comma, and $-prefixed forms.
  - Dataset builds and yields a valid EnvGroupBuilder; the env produces a
    well-formed ModelInput and a zero-reward StepResult for a dummy action.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.multipersona_gsm8k import (  # noqa: E402
    PROMPT_TEMPLATE,
    MultipersonaGsm8kDatasetBuilder,
    MultipersonaGsm8kEnv,
    check_correct,
    check_tag_structure,
    extract_answer_text,
    extract_gsm8k_gold,
)


def red(msg): return f"\033[31m{msg}\033[0m"
def green(msg): return f"\033[32m{msg}\033[0m"


failures = 0

def check(name, cond, detail=""):
    global failures
    if cond:
        print(f"  {green('PASS')} {name}")
    else:
        print(f"  {red('FAIL')} {name}  {detail}")
        failures += 1


# -------------------------------------------------------------------------
# 1. tag_structure
# -------------------------------------------------------------------------
print("== tag_structure ==")
canonical = "<mutipersonaDebate>Persona A says X; Persona B says Y.</mutipersonaDebate> <answer>42</answer>"
check("canonical passes", check_tag_structure(canonical))

unclosed_answer = "<mutipersonaDebate>debate</mutipersonaDebate> <answer>7"
check("unclosed <answer> passes (stop-sequence stripped)", check_tag_structure(unclosed_answer))

no_debate = "<answer>42</answer>"
check("no debate tag fails", not check_tag_structure(no_debate))

empty_debate = "<mutipersonaDebate>  </mutipersonaDebate> <answer>42</answer>"
check("empty debate fails", not check_tag_structure(empty_debate))

empty_answer = "<mutipersonaDebate>stuff</mutipersonaDebate> <answer>   </answer>"
check("empty answer fails", not check_tag_structure(empty_answer))

reversed_order = "<answer>42</answer> <mutipersonaDebate>debate</mutipersonaDebate>"
check("reversed order fails", not check_tag_structure(reversed_order))

wrong_spelling = "<multipersonaDebate>x</multipersonaDebate> <answer>42</answer>"
check("wrong spelling (correct 'multi-') fails", not check_tag_structure(wrong_spelling))


# -------------------------------------------------------------------------
# 2. answer extraction
# -------------------------------------------------------------------------
print("\n== extract_answer_text ==")
check("closed block", extract_answer_text(canonical) == "42")
check("unclosed block", extract_answer_text(unclosed_answer) == "7")
check("boxed preserved", extract_answer_text(
    "<mutipersonaDebate>x</mutipersonaDebate> <answer>\\boxed{12}</answer>"
) == "\\boxed{12}")
check("no answer tag → None", extract_answer_text("just some text") is None)


# -------------------------------------------------------------------------
# 3. GSM8K gold extraction
# -------------------------------------------------------------------------
print("\n== extract_gsm8k_gold ==")
gsm_answer = ("Let's compute step 1: 5+3=8.\n"
              "Step 2: 8*2=16.\n"
              "#### 16")
check("standard #### line", extract_gsm8k_gold(gsm_answer) == "16")

gsm_comma = "a lot of work\n#### 1,200"
check("comma-separated", extract_gsm8k_gold(gsm_comma) == "1200")


# -------------------------------------------------------------------------
# 4. check_correct (numeric grading)
# -------------------------------------------------------------------------
print("\n== check_correct ==")
c = lambda comp, g: check_correct(comp, g)
check("exact int match", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>42</answer>", "42"))
check("comma normalized", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>1,200</answer>", "1200"))
check("dollar prefix", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>$18</answer>", "18"))
check("float = int", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>3.0</answer>", "3"))
check("boxed wrapped", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>\\boxed{8}</answer>", "8"))
check("trailing period", c("<mutipersonaDebate>x</mutipersonaDebate> <answer>8.</answer>", "8"))
check("wrong answer rejected", not c("<mutipersonaDebate>x</mutipersonaDebate> <answer>9</answer>", "8"))
check("no answer → False", not c("some garbage", "5"))
check("text with embedded number: 'the answer is 12'",
      c("<mutipersonaDebate>x</mutipersonaDebate> <answer>the answer is 12</answer>", "12"))


# -------------------------------------------------------------------------
# 5. Prompt template formatting
# -------------------------------------------------------------------------
print("\n== PROMPT_TEMPLATE ==")
sample = PROMPT_TEMPLATE.format(problem="Alice has 3 apples. Bob has 5.")
check("ends with 'Assistant: '", sample.endswith("Assistant: "))
check("contains the exact debate tag spelling", "<mutipersonaDebate>" in sample)
check("contains the <answer> placeholder description", "<answer>answer here </answer>" in sample)
check("contains the problem", "Alice has 3 apples" in sample)


# -------------------------------------------------------------------------
# 6. Build env from dataset, run a dummy step
# -------------------------------------------------------------------------
print("\n== dataset build + dummy step ==")

async def _build_and_step():
    builder = MultipersonaGsm8kDatasetBuilder(
        batch_size=4,
        model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Base",
        group_size=2,
        seed=20260423,
        n_train=8,
        n_test=4,
    )
    train, test = await builder()
    print(f"  train batches: {len(train)}, test batches: {len(test)}")
    batch = train.get_batch(0)
    print(f"  first batch: {len(batch)} group builders")
    envs = await batch[0].make_envs()
    print(f"  first group: {len(envs)} envs (group_size)")

    env = envs[0]
    obs, stop = await env.initial_observation()
    n_tok = obs.length if hasattr(obs, "length") else len(obs.to_ints())
    print(f"  initial_observation: {n_tok} tokens, stop={stop}")
    print(f"  gold: {env.gold}")

    # Dummy action: simulate a bad response (empty)
    result = await env.step([], extra={"stop_reason": "length"})
    print(f"  step with empty action → reward={result.reward:.3f}  done={result.episode_done}")
    return result, env

result, env = asyncio.run(_build_and_step())
check("episode_done on empty action", result.episode_done)
check("reward on empty action is -tag_coef", abs(result.reward - (-0.2)) < 1e-9)

# Also test a fully-correct synthetic completion
print("\n== synthetic correct completion ==")
async def _correct():
    builder = MultipersonaGsm8kDatasetBuilder(
        batch_size=1, model_name_for_tokenizer="Qwen/Qwen3-30B-A3B-Base",
        group_size=1, seed=0, n_train=1, n_test=1,
    )
    train, _ = await builder()
    env = (await train.get_batch(0)[0].make_envs())[0]
    synthetic = (f"<mutipersonaDebate>We analyze. Alice: let's compute. "
                 f"Bob: the answer is {env.gold}.</mutipersonaDebate> "
                 f"<answer>{env.gold}</answer>")
    toks = env.tokenizer.encode(synthetic, add_special_tokens=False)
    r = await env.step(toks, extra={"stop_reason": "stop"})
    print(f"  synthetic-correct reward={r.reward:.3f}  correct={r.metrics['correct']}  tag_valid={r.metrics['tag_valid']}")
    return r

rr = asyncio.run(_correct())
check("synthetic-correct reward == 1.0", abs(rr.reward - 1.0) < 1e-9)
check("correct metric == 1.0", rr.metrics["correct"] == 1.0)
check("tag_valid metric == 1.0", rr.metrics["tag_valid"] == 1.0)


# -------------------------------------------------------------------------
print()
if failures:
    print(red(f"FAILED {failures} check(s)"))
    sys.exit(1)
print(green("All checks passed."))
