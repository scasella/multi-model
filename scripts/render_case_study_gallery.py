"""Read reports/case_study/transcripts.json and render a static gallery
HTML page that compares the panel checkpoint against vanilla
Qwen3-30B-A3B-Thinking side-by-side.

The output style follows reports/blog_post/diversity.html: warm cream
background, Libre Baskerville serif body, Inter sans labels, the same
header link chips and design tokens. Each problem is rendered as a
section with the prompt up top and two columns below — panel reasoning
+ answer on the left, thinking reasoning + answer on the right. Long
reasoning bodies collapse into <details> blocks that expand on click.

The "tests" and "expected" fields from the source JSON drive a per-
problem caption box so a reader can see at a glance what the problem
was probing for.

Usage:
    python scripts/render_case_study_gallery.py
    python scripts/render_case_study_gallery.py \
        --input reports/case_study/transcripts.json \
        --output reports/case_study/gallery.html
"""
from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


HTML_TEMPLATE_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Side-by-side: panel vs Qwen3-thinking | Stephen Casella</title>
  <meta name="description" content="A static gallery comparing one sample from a panel-of-experts checkpoint against one sample from vanilla Qwen3-30B-A3B-Thinking on twenty reasoning problems across nine categories." />
  <meta name="author" content="Stephen Casella" />
  <meta name="theme-color" content="#F9F9F6" />
  <meta property="og:site_name" content="Stephen Casella" />
  <meta property="og:title" content="Side-by-side: panel vs Qwen3-thinking" />
  <meta property="og:description" content="One sample, twenty problems, two reasoning scaffolds rendered next to each other." />
  <meta property="og:type" content="article" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg:           #F9F9F6;
      --accent:       #090909;
      --fg:           #3B3B39;
      --fg-soft:      #3B3B39;
      --fg-mute:      #595856;
      --fg-faint:     #969592;
      --on-dark:      #DFDEDA;
      --surface-muted:#EFEFEB;
      --grid:         #E4E3DE;
      --teal:         #5C8A95;
      --mauve:        #8B6F8C;
      --warn:         #C18549;
      --font-serif:   'Libre Baskerville', Georgia, 'Times New Roman', serif;
      --font-sans:    'Inter', 'Helvetica Neue', Arial, sans-serif;
    }
    *,*::before,*::after { margin: 0; padding: 0; box-sizing: border-box; }
    html {
      background: var(--bg);
      overflow-y: scroll;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-rendering: optimizeLegibility;
      font-kerning: normal;
      font-variant-ligatures: common-ligatures contextual;
      font-variant-numeric: oldstyle-nums proportional-nums;
    }
    body {
      min-height: 100vh;
      font-family: var(--font-serif);
      background: var(--bg);
      color: var(--accent);
      display: flex;
      flex-direction: column;
      text-wrap: pretty;
      overflow-x: hidden;
    }
    h1, h2, h3, .subtitle { text-wrap: balance; }
    .page-surface { background: var(--bg); display: flex; flex: 1 0 auto; flex-direction: column; width: 100%; }
    .main { flex: 1; }

    .perspectives-header {
      display: flex; flex-direction: row; justify-content: space-between; align-items: center;
      padding: 2rem 2.5rem; width: 100%;
    }
    .header-link {
      display: inline-flex; align-items: center; justify-content: center;
      padding: 0.5rem 0.75rem; border-radius: 999px; color: var(--fg);
      transition: background 240ms cubic-bezier(0.4, 0, 0.2, 1);
      text-decoration: none; font-family: var(--font-sans); font-size: 13px;
      letter-spacing: 0.08em; text-transform: uppercase;
    }
    .header-link:first-child { margin-left: -0.75rem; }
    .header-link:last-child { margin-right: -0.75rem; }
    .header-link:hover, .header-link:focus-visible { background: var(--surface-muted); }

    .benchmarks-content {
      max-width: 720px; margin: 0 auto; padding: 2rem 2rem 2rem;
    }
    .benchmarks-content h1 {
      font-size: 48px; font-weight: 400; line-height: 1.1;
      letter-spacing: -0.01em; color: var(--fg); margin-top: 144px;
    }
    .benchmarks-content .subtitle {
      font-size: 18px; font-weight: 300; line-height: 1.5;
      color: var(--fg); margin-top: 10px;
    }
    .benchmarks-content .date {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.1em;
      color: var(--fg-faint); margin-top: 1em; text-transform: uppercase;
    }
    .paper-links { display: flex; gap: 1rem; margin: 26px 0; flex-wrap: wrap; }
    .btn-pill {
      font-family: var(--font-sans); font-size: 14px; line-height: 1; color: var(--fg);
      text-decoration: none; background: #fff; border-radius: 999px;
      padding: 0.85rem 1.2rem 0.95rem;
      transition: background 240ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    .btn-pill:hover { background: var(--surface-muted); }

    .benchmarks-body { margin: 64px 0 48px; }
    .benchmarks-body p { font-weight: 400; color: var(--fg-soft); margin-bottom: 1lh; line-height: 1.6; font-size: 17px; }
    .benchmarks-body strong { font-weight: 500; color: var(--fg); }
    .benchmarks-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.82em; background: rgba(0,0,0,0.05);
      padding: 0.08em 0.3em; border-radius: 4px;
    }

    .tldr {
      margin: 32px 0 48px; padding: 0 0 0 24px;
      border-left: 2px solid var(--fg-mute);
    }
    .tldr-label {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.18em;
      text-transform: uppercase; color: var(--fg-faint);
      margin-bottom: 10px; display: block;
    }
    .tldr p { font-size: 17px; font-weight: 400; line-height: 1.6; color: var(--fg); }

    /* Gallery */
    .gallery {
      max-width: 1280px; margin: 0 auto; padding: 0 2rem 5rem;
    }
    .gallery-intro {
      max-width: 720px; margin: 0 auto 3rem;
      font-size: 17px; line-height: 1.6; color: var(--fg-soft); font-weight: 400;
    }
    .category-section {
      margin: 5rem 0 3rem;
      max-width: 1280px;
    }
    .category-header {
      max-width: 720px; margin: 0 auto 2rem;
    }
    .category-tag {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.16em;
      text-transform: uppercase; color: var(--fg-faint); display: block; margin-bottom: 8px;
    }
    .category-header h2 {
      font-size: 30px; font-weight: 400; color: var(--fg); letter-spacing: -0.005em;
      line-height: 1.15;
    }
    .category-header p {
      font-size: 16px; line-height: 1.55; color: var(--fg-mute); font-weight: 400; margin-top: 0.6rem;
    }

    .problem {
      margin: 2.5rem 0;
      border-top: 1px solid var(--grid);
      padding-top: 2rem;
    }
    .problem-id {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.16em;
      text-transform: uppercase; color: var(--fg-faint); margin-bottom: 8px;
    }
    .problem-text {
      font-size: 19px; font-weight: 400; line-height: 1.45; color: var(--fg);
      margin-bottom: 1.25rem; font-style: italic;
    }
    .problem-meta {
      font-family: var(--font-sans); font-size: 12.5px; line-height: 1.55;
      color: var(--fg-mute); padding: 0.85rem 1rem; background: var(--surface-muted);
      border-radius: 8px; margin-bottom: 1.5rem;
    }
    .problem-meta strong { color: var(--fg); font-weight: 500; }

    .answer-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }
    @media (max-width: 720px) { .answer-grid { grid-template-columns: 1fr; } }
    .answer-card {
      background: #fff; border: 1px solid var(--grid); border-radius: 12px;
      padding: 1.25rem 1.25rem 1.4rem; display: flex; flex-direction: column;
    }
    .answer-card.panel  { border-top: 3px solid var(--teal); }
    .answer-card.thinking { border-top: 3px solid var(--mauve); }
    .arm-label {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.14em;
      text-transform: uppercase; color: var(--fg-faint); margin-bottom: 6px;
    }
    .arm-label .pill {
      font-family: var(--font-sans); font-size: 10px;
      padding: 1px 8px; border-radius: 999px; margin-left: 8px;
      letter-spacing: 0.06em;
    }
    .answer-card.panel .arm-label .pill { background: rgba(92,138,149,0.15); color: var(--teal); }
    .answer-card.thinking .arm-label .pill { background: rgba(139,111,140,0.15); color: var(--mauve); }
    .arm-name {
      font-family: var(--font-serif); font-size: 17px; font-weight: 700; color: var(--fg);
      margin-bottom: 0.85rem;
    }
    .answer-final {
      font-family: var(--font-serif); font-size: 18px; font-weight: 400;
      line-height: 1.45; color: var(--fg);
      padding: 0.85rem 1rem; border-radius: 8px;
      background: var(--bg); border: 1px solid var(--grid); margin-bottom: 1rem;
      white-space: pre-wrap; word-break: break-word;
    }
    .answer-final.empty { color: var(--fg-faint); font-style: italic; }
    details.reasoning {
      margin-top: 0.5rem; border-top: 1px dashed var(--grid); padding-top: 0.75rem;
    }
    details.reasoning summary {
      font-family: var(--font-sans); font-size: 12px; letter-spacing: 0.06em;
      color: var(--fg-mute); cursor: pointer; user-select: none;
      list-style: none; padding: 0.25rem 0;
    }
    details.reasoning summary::-webkit-details-marker { display: none; }
    details.reasoning summary::before {
      content: "▸ "; color: var(--fg-faint); transition: transform 150ms;
      display: inline-block;
    }
    details[open].reasoning summary::before { content: "▾ "; }
    .reasoning-body {
      font-family: var(--font-sans); font-size: 13px; line-height: 1.6;
      color: var(--fg-soft); margin-top: 0.6rem; white-space: pre-wrap; word-break: break-word;
      max-height: 320px; overflow-y: auto;
      padding: 0.75rem 0.85rem; background: var(--surface-muted); border-radius: 6px;
      font-variant-numeric: tabular-nums;
    }
    .reasoning-body::-webkit-scrollbar { width: 8px; }
    .reasoning-body::-webkit-scrollbar-thumb { background: var(--fg-faint); border-radius: 4px; }
    .reasoning-stats {
      font-family: var(--font-sans); font-size: 11px; color: var(--fg-faint);
      margin-top: 0.5rem;
    }

    .footnote {
      max-width: 720px; margin: 5rem auto 0; padding: 2rem;
      border-top: 1px solid var(--grid);
      font-family: var(--font-sans); font-size: 12px; line-height: 1.6; color: var(--fg-faint);
    }
    .footnote a { color: var(--fg); border-bottom: 1px dotted var(--fg-faint); text-decoration: none; }
    .footnote a:hover { border-bottom-color: var(--fg); }

    @media (max-width: 720px) {
      .perspectives-header { padding: 1.5rem; }
      .benchmarks-content { padding: 2rem 1.25rem; }
      .benchmarks-content h1 { font-size: 36px; margin-top: 80px; }
      .gallery { padding: 0 1rem 4rem; }
      .category-section { margin: 3rem 0 2rem; }
      .category-header h2 { font-size: 24px; }
      .problem-text { font-size: 17px; }
      .answer-card { padding: 1rem; }
    }
  </style>
</head>
<body class="benchmarks-page">
  <div class="page-surface">
"""


def render(items: list[dict], meta: dict) -> str:
    # Group items by category, preserving order
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for it in items:
        grouped.setdefault(it["category"], []).append(it)

    panel_label = "panel adapter (LoRA r32 on Qwen3-30B-A3B-Base)"
    thinking_label = "Qwen3-30B-A3B (native thinking)"

    out = [HTML_TEMPLATE_HEAD]

    out.append("""
    <header class="perspectives-header">
      <a href="../blog_post/diversity.html" class="header-link">Back to writeup</a>
      <a href="https://github.com/scasella/multi-model" class="header-link">GitHub</a>
    </header>

    <main class="main">
      <article class="benchmarks-content">

        <h1>Side-by-side: panel vs Qwen3-thinking</h1>
        <p class="subtitle">Twenty reasoning problems, sampled once each from the published panel checkpoint and from vanilla Qwen3-30B-A3B-Thinking, rendered next to each other.</p>
        <p class="date">STEPHEN CASELLA · APRIL 2026</p>

        <div class="paper-links">
          <a href="../blog_post/diversity.html" class="btn-pill">Read the writeup</a>
          <a href="https://github.com/scasella/multi-model" class="btn-pill">GitHub repo</a>
        </div>

        <div class="benchmarks-body">

          <div class="tldr">
            <span class="tldr-label">How to read this page</span>
            <p>
              Each row below shows one problem and one fresh sample from each scaffold at <strong>temperature 1.0</strong> &mdash; not best-of-N, not retrieval, just a single shot. Click <strong>show reasoning</strong> on either card to expand the full <code>&lt;mutipersonaDebate&gt;</code> body or the full <code>&lt;think&gt;</code> body. The point isn't to crown a winner per row: it's to let the difference in <em>shape</em> of the two reasoning traces speak for itself.
            </p>
            <p>
              The panel often arrives at the same answer with a fraction of the tokens; thinking often takes a longer route and lands somewhere different. On the underspecified, paradoxical, and counterfactual problems both arms can fail interestingly &mdash; the brackets matter more than the bottom line.
            </p>
          </div>
""")

    # Per-category sections
    for cat_idx, (cat, group) in enumerate(grouped.items()):
        first = group[0]
        out.append(f"""
        </div>
      </article>
    </main>

    <section class="gallery">
      <div class="category-section">
        <div class="category-header">
          <span class="category-tag">Category {cat_idx + 1:02d}</span>
          <h2>{html.escape(cat)}</h2>
""")
        if first.get("category_blurb"):
            out.append(f"          <p>{html.escape(first['category_blurb'])}</p>\n")
        out.append("""        </div>
""")

        for it in group:
            panel = it.get("panel", {})
            think = it.get("thinking", {})
            panel_reasoning = panel.get("reasoning", "")
            panel_answer = panel.get("answer", "") or ""
            think_reasoning = think.get("reasoning", "")
            think_answer = think.get("answer", "") or ""

            panel_chars = len(panel.get("raw", "") or "")
            think_chars = len(think.get("raw", "") or "")

            problem_html = html.escape(it["problem"])
            tests_html = html.escape(it.get("tests", ""))
            expected_html = html.escape(it.get("expected", ""))

            panel_answer_block = (
                f'<div class="answer-final">{html.escape(panel_answer)}</div>'
                if panel_answer.strip()
                else '<div class="answer-final empty">(no &lt;answer&gt; tag emitted)</div>'
            )
            think_answer_block = (
                f'<div class="answer-final">{html.escape(think_answer)}</div>'
                if think_answer.strip()
                else '<div class="answer-final empty">(no post-think output)</div>'
            )

            panel_reasoning_block = (
                f'<details class="reasoning"><summary>show reasoning ({panel_chars:,} chars)</summary>'
                f'<div class="reasoning-body">{html.escape(panel_reasoning)}</div>'
                '</details>'
            ) if panel_reasoning.strip() else ""

            think_reasoning_block = (
                f'<details class="reasoning"><summary>show reasoning ({think_chars:,} chars)</summary>'
                f'<div class="reasoning-body">{html.escape(think_reasoning)}</div>'
                '</details>'
            ) if think_reasoning.strip() else ""

            out.append(f"""
        <div class="problem" id="prob-{html.escape(it['id'])}">
          <div class="problem-id">Problem {html.escape(it['id'])}</div>
          <div class="problem-text">{problem_html}</div>
          <div class="problem-meta">
            <strong>Tests:</strong> {tests_html}<br />
            <strong>A reasonable answer:</strong> {expected_html}
          </div>
          <div class="answer-grid">
            <div class="answer-card panel">
              <div class="arm-label">arm <span class="pill">panel</span></div>
              <div class="arm-name">{html.escape(panel_label)}</div>
              {panel_answer_block}
              {panel_reasoning_block}
            </div>
            <div class="answer-card thinking">
              <div class="arm-label">arm <span class="pill">thinking</span></div>
              <div class="arm-name">{html.escape(thinking_label)}</div>
              {think_answer_block}
              {think_reasoning_block}
            </div>
          </div>
        </div>
""")

        out.append("      </div>\n    </section>\n")

    # Footnote with metadata
    out.append(f"""
    <div class="footnote">
      <p>
        Sampled at temperature {meta.get('temperature', 1.0)} with <code>max_tokens={meta.get('max_tokens', '—')}</code>; one sample per problem per arm. The panel side hits
        <code>{html.escape(meta.get('panel_checkpoint', ''))}</code>; the thinking side hits the public <code>{html.escape(meta.get('thinking_backbone', ''))}</code> via <code>apply_chat_template(enable_thinking=True)</code>.
        Generated {html.escape(meta.get('produced_at', ''))} from <code>{html.escape(meta.get('panel_backbone', ''))}</code>. Source: <a href="https://github.com/scasella/multi-model/blob/case-study/scripts/build_case_study_transcripts.py">build_case_study_transcripts.py</a>.
      </p>
    </div>
""")

    out.append("""
  </div>
</body>
</html>
""")
    return "".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(ROOT / "reports" / "case_study" / "transcripts.json"))
    p.add_argument("--output", default=str(ROOT / "reports" / "case_study" / "gallery.html"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    items = data.get("items", [])
    meta = data.get("metadata", {})
    rendered = render(items, meta)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered)
    print(f"  wrote {out}  ({out.stat().st_size/1024:.1f} KB · {len(items)} items)")


if __name__ == "__main__":
    main()
