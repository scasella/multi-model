"""Sample 20 reasoning problems against both arms (panel checkpoint + vanilla
Qwen3-30B-A3B thinking) and write the transcripts to JSON for the gallery.

Each problem hits each arm exactly once at temperature=1.0, so what you see
is one representative sample, not a mode-collapsed best-of-N. The 20 problems
are organised into 9 categories that probe different reasoning capabilities:
two-route convergence, distractor handling, underspecified prompts, self-
reference, multi-step word problems, decision/game theory, cross-domain
transfer, Fermi estimation, and counterfactual/causal.

The panel checkpoint defaults to the published post-MATH-RL session referenced
as eval session 44722365 in reports/blog_post/diversity.html. Override with
PANEL_MATH_CHECKPOINT_SAMPLER. The thinking arm uses the public
Qwen/Qwen3-30B-A3B production model via apply_chat_template(enable_thinking=True).

Usage:
    python scripts/build_case_study_transcripts.py
    python scripts/build_case_study_transcripts.py --max-tokens 6000
    python scripts/build_case_study_transcripts.py --output reports/case_study/transcripts.json

Output schema:
    {"items": [
        {"id": "a", "category": "...", "category_short": "...", "problem": "...",
         "tests": "...", "expected": "...",
         "panel": {"reasoning": "...", "answer": "...", "raw": "..."},
         "thinking": {"reasoning": "...", "answer": "...", "raw": "..."}},
        ...
    ], "metadata": {...}}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Fallback for editable-install path that gets cleared from /private/tmp:
for cand in ("/Users/scasella/Downloads/tinker-cookbook-main",):
    if Path(cand).is_dir():
        sys.path.insert(0, cand)

import tinker  # noqa: E402
from tinker_cookbook.completers import TinkerTokenCompleter  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from envs.multipersona_gsm8k import (  # noqa: E402
    PROMPT_TEMPLATE as PANEL_PROMPT,
    STOP_SEQUENCES as PANEL_STOP,
    extract_answer_text,
)


PANEL_BACKBONE = "Qwen/Qwen3-30B-A3B-Base"
THINKING_BACKBONE = "Qwen/Qwen3-30B-A3B"
THINKING_STOP = ["<|im_end|>"]

DEFAULT_PANEL_CKPT = os.environ.get(
    "PANEL_MATH_CHECKPOINT_SAMPLER",
    "tinker://44722365-baf6-517f-a69f-ec01d13f6395:train:0/sampler_weights/final",
)


# ---------------------------------------------------------------------------
# Problem set — 20 items across 9 categories. All wording is original (not
# copied from MATH/AIME/GSM8K/OlympiadBench).
# ---------------------------------------------------------------------------
PROBLEMS = [
    # --- 1. Two-route convergence ----------------------------------------
    {
        "id": "a", "category": "Two-route convergence",
        "category_short": "two_route",
        "category_blurb": "Problems with two natural solution paths. A multi-persona debate that's working should produce two routes that meet at the same answer.",
        "problem": "A circle passes through the points (0,0), (8,0), and (0,6). What is the radius?",
        "tests": "Whether the persona structure surfaces two distinct methods (perpendicular bisectors vs. general circle equation) before converging.",
        "expected": "r = 5.",
    },
    {
        "id": "b", "category": "Two-route convergence",
        "category_short": "two_route",
        "category_blurb": None,
        "problem": "Three fair six-sided dice are rolled. What is the probability that the maximum value showing is exactly 4?",
        "tests": "Inclusion-exclusion (\"max ≤ 4 minus max ≤ 3\") vs. direct enumeration over all 216 outcomes.",
        "expected": "37/216 ≈ 0.171.",
    },
    {
        "id": "c", "category": "Two-route convergence",
        "category_short": "two_route",
        "category_blurb": None,
        "problem": "On a 5x5 grid of squares, in how many ways can you place 3 non-attacking rooks (rooks attack along rows and columns)?",
        "tests": "Direct counting (choose 3 rows × choose 3 cols × permute) vs. inclusion-exclusion.",
        "expected": "1000.",
    },
    # --- 2. Distractor / misleading-surface ------------------------------
    {
        "id": "d", "category": "Distractor / misleading surface",
        "category_short": "distractor",
        "category_blurb": "Problems with extra information that's irrelevant, or wording that misleads the surface reader.",
        "problem": "Sarah has $50. She buys 3 paperback books at $8 each, then her friend gives her $10 in birthday money. Each book is 320 pages long and the bookstore closes at 6pm. How much money does Sarah have now?",
        "tests": "Whether the irrelevant page count and closing time get silently ignored, or wastefully addressed.",
        "expected": "$36.",
    },
    {
        "id": "e", "category": "Distractor / misleading surface",
        "category_short": "distractor",
        "category_blurb": None,
        "problem": "Three workers can paint three rooms in three hours, working at the same rate. How long would 100 workers take to paint 100 rooms?",
        "tests": "Whether the model resists the surface-symmetry trap and computes the per-worker per-room rate.",
        "expected": "3 hours (the rate is one room per worker per three hours).",
    },
    {
        "id": "f", "category": "Distractor / misleading surface",
        "category_short": "distractor",
        "category_blurb": None,
        "problem": "A train leaves at noon traveling east at 60 mph. At the same moment, a bird at the station 90 miles east starts flying west at 120 mph. The bird turns around the moment it meets the train. How long is the bird in the air?",
        "tests": "Whether the model collapses the convoluted setup to a simple relative-speed problem.",
        "expected": "0.5 hours (90 miles / (60+120) mph).",
    },
    # --- 3. Underspecified / no-clean-answer -----------------------------
    {
        "id": "g", "category": "Underspecified / calibration",
        "category_short": "underspec",
        "category_blurb": "Prompts where the right answer is to acknowledge missing information rather than commit to a number.",
        "problem": "I bought a dozen apples this morning and ate some at lunch. How many apples do I have left?",
        "tests": "Calibration. Healthy: notes the missing information and parametrises. Pathological: invents a number.",
        "expected": "Underspecified — anywhere between 0 and 12 depending on how many were eaten.",
    },
    {
        "id": "h", "category": "Underspecified / calibration",
        "category_short": "underspec",
        "category_blurb": None,
        "problem": "What is half of two plus two?",
        "tests": "Whether the ambiguity (half of (2+2) vs. (half of 2)+2) is called out.",
        "expected": "Genuinely ambiguous — 2 or 3 depending on grouping.",
    },
    {
        "id": "i", "category": "Underspecified / calibration",
        "category_short": "underspec",
        "category_blurb": None,
        "problem": "Find an integer between 5 and 9.",
        "tests": "Whether the multi-valued answer is acknowledged.",
        "expected": "Any of 6, 7, 8 (or 5 and 9 inclusive depending on interpretation).",
    },
    # --- 4. Self-reference / impossibility -------------------------------
    {
        "id": "j", "category": "Self-reference / impossibility",
        "category_short": "selfref",
        "category_blurb": "Problems where the right answer is detecting an impossibility, not forcing a number.",
        "problem": "A says: \"I am a liar.\" Assume A is either a perfectly honest truth-teller or a consistent liar. What is A?",
        "tests": "Paradox detection rather than forced commitment.",
        "expected": "Neither — the statement is paradoxical under both classifications.",
    },
    {
        "id": "k", "category": "Self-reference / impossibility",
        "category_short": "selfref",
        "category_blurb": None,
        "problem": "This problem has exactly N letters in the word for its answer. Find N.",
        "tests": "Self-referential consistency — does the model land on the fixed point?",
        "expected": "N = 4 (the word \"four\" has four letters).",
    },
    # --- 5. Multi-step word with off-by-one ------------------------------
    {
        "id": "l", "category": "Multi-step word problems",
        "category_short": "multistep",
        "category_blurb": "Problems where naive iteration overshoots; a careful reader catches the off-by-one.",
        "problem": "A snail is at the bottom of a 10-meter well. Each day it climbs up 3 meters, but each night it slides back down 2 meters. How many days until it reaches the top of the well?",
        "tests": "Whether the model notices the snail doesn't slide back the day it reaches the top.",
        "expected": "8 days.",
    },
    {
        "id": "m", "category": "Multi-step word problems",
        "category_short": "multistep",
        "category_blurb": None,
        "problem": "A 3-digit number has these properties: the hundreds digit is twice the tens digit; the tens digit is three more than the units digit; the digits sum to 13. What is the number?",
        "tests": "Substitution algebra plus the digit-validity constraint H ≤ 9.",
        "expected": "841.",
    },
    # --- 6. Game theory + decision -------------------------------------------
    {
        "id": "n", "category": "Game theory and decision",
        "category_short": "game",
        "category_blurb": "Problems that reward deriving an invariant and noticing the surprising consequence.",
        "problem": "Two players alternate removing 1, 2, or 3 stones from a pile of 20. Whoever takes the last stone wins. With perfect play, who wins, and what is the first player's first move?",
        "tests": "Whether the mod-4 invariant is derived; the surprise is that the first player loses.",
        "expected": "Second player wins (20 ≡ 0 mod 4); the first player has no winning move.",
    },
    {
        "id": "o", "category": "Game theory and decision",
        "category_short": "game",
        "category_blurb": None,
        "problem": "You're offered a choice: (i) $100 right now, or (ii) flip a fair coin and get $250 on heads, $0 on tails. Which has the higher expected value, and which would you actually take? Justify the second answer.",
        "tests": "Whether the model separates expected value from utility (risk aversion).",
        "expected": "EV(coin) = $125 > $100; but the rational pick depends on risk preferences and the marginal utility of $100 vs $250.",
    },
    # --- 7. Cross-domain transfer ----------------------------------------
    {
        "id": "p", "category": "Cross-domain transfer",
        "category_short": "transfer",
        "category_blurb": "Problems that bridge math and physics, or math and language; tests whether the math-trained model carries skill across domains.",
        "problem": "If Earth's gravity suddenly doubled, what would happen to the period of a grandfather clock? Give both the qualitative direction and the multiplicative factor.",
        "tests": "Whether the model retrieves T = 2π√(L/g) and applies it cleanly.",
        "expected": "Period shrinks by a factor of √2; the clock ticks faster (period multiplied by 1/√2 ≈ 0.707).",
    },
    {
        "id": "q", "category": "Cross-domain transfer",
        "category_short": "transfer",
        "category_blurb": None,
        "problem": "I have two children. At least one of them is a boy who was born on a Tuesday. Given only this information, what is the probability that both children are boys?",
        "tests": "Whether the 14×14 sample-space construction is done carefully (Tuesday boy paradox).",
        "expected": "13/27 ≈ 0.481, not 1/2.",
    },
    # --- 8. Fermi estimation -------------------------------------------------
    {
        "id": "r", "category": "Fermi estimation",
        "category_short": "fermi",
        "category_blurb": "No closed-form answer. The right behaviour is a calibrated Fermi decomposition with explicit assumptions.",
        "problem": "Roughly how many haircuts does an average full-time hairstylist give in a year? Show the Fermi-style decomposition.",
        "tests": "Whether the response states assumptions and composes them transparently.",
        "expected": "Order ~2000 (e.g. 8 customers/day × 250 working days). The number is less important than the decomposition.",
    },
    {
        "id": "s", "category": "Fermi estimation",
        "category_short": "fermi",
        "category_blurb": None,
        "problem": "Estimate the total mass of all the air inside a typical home. Show your steps.",
        "tests": "Volume × density composition with unit awareness.",
        "expected": "~150 m³ × 1.225 kg/m³ ≈ 180 kg as one reasonable answer.",
    },
    # --- 9. Counterfactual / causal --------------------------------------
    {
        "id": "t", "category": "Counterfactual / causal",
        "category_short": "causal",
        "category_blurb": "Tests whether the model resists the correlation-causation slide and surfaces alternative explanations.",
        "problem": "A small town's population has been steady at 10,000 for a decade. The town opens a new public park. Over the next year, population rises to 10,500 and the local rate of new cancer diagnoses falls by 8%. Can you conclude the park caused the cancer-rate drop? Lay out your reasoning.",
        "tests": "Whether confounders (in-migration changing demographics, regression to the mean, reporting changes) are surfaced.",
        "expected": "No; correlation isn't causation; multiple plausible confounds.",
    },
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


async def sample_panel(policy, tokenizer, problem: str) -> str:
    prompt = PANEL_PROMPT.format(problem=problem)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    result = await policy(model_input, PANEL_STOP)
    return tokenizer.decode(list(result.tokens))


async def sample_thinking(policy, tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True, tokenize=False
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(list(tokens))
    result = await policy(model_input, THINKING_STOP)
    return tokenizer.decode(list(result.tokens))


# ---------------------------------------------------------------------------
# Output extraction
# ---------------------------------------------------------------------------


def split_panel(raw: str) -> tuple[str, str | None]:
    """Return (debate_body, answer_body or None)."""
    debate = raw
    if "<answer>" in debate:
        debate = debate.split("<answer>", 1)[0]
    debate = debate.strip()
    # Strip the leading <mutipersonaDebate> tag for cleaner display.
    if debate.startswith("<mutipersonaDebate>"):
        debate = debate[len("<mutipersonaDebate>"):]
    if debate.endswith("</mutipersonaDebate>"):
        debate = debate[: -len("</mutipersonaDebate>")]
    debate = debate.strip()
    answer = extract_answer_text(raw)
    return debate, answer


_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def split_thinking(raw: str) -> tuple[str, str]:
    """Return (think_body, post_think_body). post_think_body is the visible answer."""
    text = raw
    # Drop any chat-template scaffolding the decode pulled in.
    text = text.replace("<|im_end|>", "").strip()
    open_m = _THINK_OPEN.search(text)
    close_m = _THINK_CLOSE.search(text)
    if open_m and close_m and close_m.start() > open_m.end():
        think = text[open_m.end(): close_m.start()].strip()
        answer = text[close_m.end():].strip()
        return think, answer
    # No <think> tags surfaced — return everything as "answer", empty think.
    return "", text


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def build(args: argparse.Namespace) -> dict:
    print(f"Panel checkpoint: {args.panel_checkpoint}")
    print(f"Thinking model:   {THINKING_BACKBONE}")
    print(f"Problems: {len(PROBLEMS)}  ·  temperature={args.temperature}  ·  max_tokens={args.max_tokens}")
    print()

    panel_tokenizer = get_tokenizer(PANEL_BACKBONE)
    thinking_tokenizer = get_tokenizer(THINKING_BACKBONE)
    sc = tinker.ServiceClient()
    panel_sampling = sc.create_sampling_client(model_path=args.panel_checkpoint)
    thinking_sampling = sc.create_sampling_client(base_model=THINKING_BACKBONE)
    panel_policy = TinkerTokenCompleter(
        sampling_client=panel_sampling, max_tokens=args.max_tokens, temperature=args.temperature
    )
    thinking_policy = TinkerTokenCompleter(
        sampling_client=thinking_sampling, max_tokens=args.max_tokens, temperature=args.temperature
    )

    items = []
    sem = asyncio.Semaphore(args.concurrency)

    async def one(p: dict) -> dict:
        async with sem:
            t0 = time.time()
            try:
                panel_raw, thinking_raw = await asyncio.gather(
                    sample_panel(panel_policy, panel_tokenizer, p["problem"]),
                    sample_thinking(thinking_policy, thinking_tokenizer, p["problem"]),
                )
            except Exception as e:
                print(f"  [{p['id']}] sampling FAILED: {type(e).__name__}: {e}")
                return {**p, "error": f"{type(e).__name__}: {e}"}
            dur = time.time() - t0
            panel_debate, panel_answer = split_panel(panel_raw)
            thinking_think, thinking_answer = split_thinking(thinking_raw)
            print(
                f"  [{p['id']}] {p['category_short']:>10}  panel={len(panel_raw):>5} chars · "
                f"think={len(thinking_raw):>5} chars · {dur:>5.1f}s"
            )
            return {
                **p,
                "panel": {"raw": panel_raw, "reasoning": panel_debate, "answer": panel_answer or ""},
                "thinking": {"raw": thinking_raw, "reasoning": thinking_think, "answer": thinking_answer},
            }

    items = await asyncio.gather(*(one(p) for p in PROBLEMS))

    return {
        "items": items,
        "metadata": {
            "panel_checkpoint": args.panel_checkpoint,
            "thinking_backbone": THINKING_BACKBONE,
            "panel_backbone": PANEL_BACKBONE,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_problems": len(PROBLEMS),
            "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel-checkpoint", default=DEFAULT_PANEL_CKPT)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--output", default=str(ROOT / "reports" / "case_study" / "transcripts.json"),
        help="Where to write the transcript JSON."
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = await build(args)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print()
    print(f"  wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
