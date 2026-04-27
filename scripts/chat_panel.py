"""Interactive chat REPL with the panel-of-experts checkpoint on Tinker.

Type a problem (math, reasoning, anything), hit Enter, and the script
samples a single completion from the panel sampler and prints the
<mutipersonaDebate> body and the extracted <answer>. Type Ctrl-D
(EOF) or 'quit' on its own line to exit.

By default it talks to the post-MATH-RL panel checkpoint — the same one
referenced in the blog as 'eval session 44722365'. Override with the
PANEL_MATH_CHECKPOINT_SAMPLER env var, or with --checkpoint-path.

Usage:
    # simplest: REPL with the published checkpoint
    python scripts/chat_panel.py

    # one-shot
    python scripts/chat_panel.py --problem "If 2x + 3 = 11, what is x?"

    # different checkpoint
    PANEL_MATH_CHECKPOINT_SAMPLER=tinker://<your-session>:train:0/sampler_weights/final \\
        python scripts/chat_panel.py

    # different sampling settings
    python scripts/chat_panel.py --temperature 0.7 --max-tokens 4096
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

from envs.multipersona_gsm8k import (
    PROMPT_TEMPLATE,
    STOP_SEQUENCES,
    extract_answer_text,
)


# Published panel-MATH checkpoint (eval session 44722365 in the blog).
DEFAULT_CHECKPOINT = os.environ.get(
    "PANEL_MATH_CHECKPOINT_SAMPLER",
    "tinker://44722365-baf6-517f-a69f-ec01d13f6395:train:0/sampler_weights/final",
)
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive REPL against a panel-of-experts checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT,
        help=f"Tinker sampler URI (default: {DEFAULT_CHECKPOINT}).",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--problem",
        default=None,
        help="One-shot mode: sample once for this problem and exit.",
    )
    p.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered prompt before each sample (useful for debugging).",
    )
    return p.parse_args()


def render(text: str) -> tuple[str, str | None]:
    """Split a raw completion into (debate body, answer body or None)."""
    debate = text
    answer = extract_answer_text(text)
    # Strip the answer tag block from the debate view if present, for readability.
    if "<answer>" in debate:
        debate = debate.split("<answer>", 1)[0].rstrip()
    return debate, answer


def print_completion(text: str) -> None:
    debate, answer = render(text)
    print()
    print("─" * 70)
    print("DEBATE")
    print("─" * 70)
    print(debate.strip())
    print()
    print("─" * 70)
    print(f"ANSWER: {answer if answer is not None else '(no <answer> tag found)'}")
    print("─" * 70)
    print()


async def sample_one(policy: TinkerTokenCompleter, tokenizer, problem: str, show_prompt: bool) -> str:
    prompt_text = PROMPT_TEMPLATE.format(problem=problem)
    if show_prompt:
        print()
        print("RENDERED PROMPT:")
        print(prompt_text)
        print()
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)
    result = await policy(model_input, STOP_SEQUENCES)
    out_tokens = list(result.tokens)
    return tokenizer.decode(out_tokens)


async def main() -> None:
    args = parse_args()

    print(f"Loading sampling client for {args.checkpoint_path}")
    print(f"Backbone: {BASE_MODEL} · temperature={args.temperature} · max_tokens={args.max_tokens}")
    print()

    tokenizer = get_tokenizer(BASE_MODEL)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.checkpoint_path)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.problem:
        text = await sample_one(policy, tokenizer, args.problem, args.show_prompt)
        print_completion(text)
        return

    print("Type a problem, then Enter. Submit with an empty line.")
    print("Ctrl-D or 'quit' to exit. 'show-prompt on' / 'show-prompt off' to toggle.")
    print()

    show_prompt = args.show_prompt
    while True:
        try:
            print(">>> ", end="", flush=True)
            lines: list[str] = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except EOFError:
            print()
            return

        problem = "\n".join(lines).strip()
        if not problem:
            continue
        if problem.lower() in {"quit", "exit", ":q"}:
            return
        if problem.lower() == "show-prompt on":
            show_prompt = True
            print("(prompt-display enabled)")
            continue
        if problem.lower() == "show-prompt off":
            show_prompt = False
            print("(prompt-display disabled)")
            continue

        try:
            text = await sample_one(policy, tokenizer, problem, show_prompt)
            print_completion(text)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except Exception as e:
            print(f"\n!! sampling error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
