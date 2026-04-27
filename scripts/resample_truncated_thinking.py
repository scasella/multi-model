"""Re-sample only the thinking-arm transcripts that hit max_tokens before
emitting `</think>`. Leaves the multi-persona side and any thinking responses
that finished cleanly untouched, so already-validated rows aren't churned.

Detection: a thinking response is "truncated" if its raw text contains
`<think>` but no matching `</think>`. The parser leaves reasoning empty in
that case, which is what the rendered gallery shows as a missing think body.

Usage:
    python scripts/resample_truncated_thinking.py
    python scripts/resample_truncated_thinking.py --max-tokens 24000
    python scripts/resample_truncated_thinking.py --dry-run

The default budget (16384) is ~2.7× the original 6000 cap and big enough
that the thinking model has finished on every problem we've tried at that
size. Pass --max-tokens explicitly if you need more headroom.
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
for cand in ("/Users/scasella/Downloads/tinker-cookbook-main",):
    if Path(cand).is_dir():
        sys.path.insert(0, cand)

import tinker  # noqa: E402
from tinker_cookbook.completers import TinkerTokenCompleter  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

THINKING_BACKBONE = "Qwen/Qwen3-30B-A3B"
THINKING_STOP = ["<|im_end|>"]

_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def split_thinking(raw: str) -> tuple[str, str]:
    text = raw.replace("<|im_end|>", "").strip()
    open_m = _THINK_OPEN.search(text)
    close_m = _THINK_CLOSE.search(text)
    if open_m and close_m and close_m.start() > open_m.end():
        return text[open_m.end(): close_m.start()].strip(), text[close_m.end():].strip()
    return "", text


def is_truncated(raw: str) -> bool:
    """True if the raw text opened <think> but never reached </think>."""
    if not raw:
        return False
    has_open = "<think>" in raw.lower()
    has_close = "</think>" in raw.lower()
    return has_open and not has_close


async def sample_thinking(policy, tokenizer, problem: str) -> str:
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=True, tokenize=False
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(list(tokens))
    result = await policy(model_input, THINKING_STOP)
    return tokenizer.decode(list(result.tokens))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "reports" / "case_study" / "transcripts.json"))
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="Print which items would be re-sampled and exit.")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    path = Path(args.input)
    data = json.loads(path.read_text())
    items = data["items"]

    truncated = [
        (i, it) for i, it in enumerate(items)
        if is_truncated((it.get("thinking") or {}).get("raw", "") or "")
    ]
    if not truncated:
        print("  no truncated thinking traces found — nothing to do.")
        return

    print("=== truncated thinking responses (will re-sample) ===")
    for i, it in truncated:
        raw_len = len((it.get("thinking") or {}).get("raw", "") or "")
        print(f"  [{it['id']}] {it['category_short']:>10}  raw was {raw_len:>6} chars (no </think>)")
    print(f"  total: {len(truncated)} items at max_tokens={args.max_tokens}")
    print()

    if args.dry_run:
        print("  --dry-run set; exiting without sampling.")
        return

    tokenizer = get_tokenizer(THINKING_BACKBONE)
    sc = tinker.ServiceClient()
    sampling = sc.create_sampling_client(base_model=THINKING_BACKBONE)
    policy = TinkerTokenCompleter(
        sampling_client=sampling, max_tokens=args.max_tokens, temperature=args.temperature
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def one(idx: int, it: dict) -> tuple[int, str]:
        async with sem:
            t0 = time.time()
            try:
                raw = await sample_thinking(policy, tokenizer, it["problem"])
            except Exception as e:
                print(f"  [{it['id']}] sampling FAILED: {type(e).__name__}: {e}")
                return idx, ""
            dur = time.time() - t0
            still_truncated = is_truncated(raw)
            note = "still truncated!" if still_truncated else "ok"
            print(f"  [{it['id']}] {it['category_short']:>10}  new raw={len(raw):>6} chars · {dur:>5.1f}s · {note}")
            return idx, raw

    results = await asyncio.gather(*(one(i, it) for i, it in truncated))

    n_patched = 0
    for idx, raw in results:
        if not raw:
            continue
        reasoning, answer = split_thinking(raw)
        items[idx]["thinking"] = {"raw": raw, "reasoning": reasoning, "answer": answer}
        n_patched += 1

    if n_patched:
        # Update metadata to reflect that some items were re-sampled at a
        # different max_tokens budget than the original build.
        meta = data.setdefault("metadata", {})
        meta.setdefault("resamples", []).append({
            "ids": [it["id"] for _, it in truncated],
            "reason": "thinking trace truncated by original max_tokens cap",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "produced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print()
        print(f"  patched {n_patched} entries in {path}")
    else:
        print("  no patches applied (all re-samples failed).")


if __name__ == "__main__":
    asyncio.run(main())
