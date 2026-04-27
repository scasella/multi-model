"""Read reports/case_study/transcripts.json and render a static gallery
HTML page that compares the multi-persona debate checkpoint against
vanilla Qwen3-30B-A3B-Thinking side-by-side.

The output style follows reports/blog_post/diversity.html: warm cream
background, Libre Baskerville serif body, Inter sans labels, the same
header link chips and design tokens. Each problem is rendered as a
section with the prompt up top and two columns below — multi-persona
debate reasoning + answer on the left, thinking reasoning + answer on
the right. Both the reasoning and the answer are always visible and
both are rendered as Markdown via marked + KaTeX from CDN.

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


def extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last `\\boxed{...}` in `text`, brace-balanced.

    Inlined from envs.multipersona_math so the renderer has no Tinker
    dependency — the env module pulls `chz` which the renderer doesn't need.
    """
    if not text:
        return None
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    start = i
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


HTML_TEMPLATE_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Side-by-side: multi-persona debate vs Qwen3-thinking | Stephen Casella</title>
  <meta name="description" content="A static gallery comparing one sample from a multi-persona debate checkpoint against one sample from vanilla Qwen3-30B-A3B-Thinking on twenty reasoning problems across nine categories." />
  <meta name="author" content="Stephen Casella" />
  <meta name="theme-color" content="#F9F9F6" />
  <meta property="og:site_name" content="Stephen Casella" />
  <meta property="og:title" content="Side-by-side: multi-persona debate vs Qwen3-thinking" />
  <meta property="og:description" content="One sample, twenty problems, two reasoning scaffolds rendered next to each other." />
  <meta property="og:type" content="article" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
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
    .section-label {
      font-family: var(--font-sans); font-size: 11px; letter-spacing: 0.14em;
      text-transform: uppercase; color: var(--fg-faint);
      margin: 1.1rem 0 0.5rem;
    }
    .section-label:first-of-type { margin-top: 0.25rem; }
    .reasoning-body, .answer-body {
      font-family: var(--font-serif);
      font-size: 14.5px;
      line-height: 1.55;
      color: var(--fg);
      padding: 0.85rem 1rem;
      border-radius: 8px;
      background: var(--surface-muted);
      word-break: break-word;
      overflow-wrap: anywhere;
    }
    .reasoning-body {
      /* Bound the visual height; scroll within for very long traces (the
         thinking arm sometimes runs >40 KB on a single problem). */
      max-height: 360px;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: var(--fg-faint) transparent;
      position: relative;
    }
    .reasoning-body::-webkit-scrollbar { width: 6px; }
    .reasoning-body::-webkit-scrollbar-thumb { background: var(--fg-faint); border-radius: 3px; }
    .reasoning-body::-webkit-scrollbar-track { background: transparent; }
    .answer-body {
      background: var(--bg);
      border: 1px solid var(--grid);
      font-size: 16px;
    }
    .answer-body.empty { color: var(--fg-faint); font-style: italic; }
    /* Section-label ratio annotation (e.g. "29× the multi-persona side").
       Only emitted when the thinking trace is materially longer than the
       multi-persona trace on the same problem, so the per-rollout token
       cost asymmetry from the writeup is legible at a glance. */
    .section-label .ratio {
      font-family: var(--font-serif);
      font-weight: 700;
      letter-spacing: 0;
      color: var(--mauve);
      text-transform: none;
      margin-left: 6px;
      padding: 1px 6px;
      background: rgba(139,111,140,0.12);
      border-radius: 4px;
      font-size: 11.5px;
    }
    /* Markdown rendering inside the reasoning + answer blocks. */
    .markdown-body p { margin: 0 0 0.7em; line-height: 1.55; }
    .markdown-body p:last-child { margin-bottom: 0; }
    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {
      font-family: var(--font-serif); font-weight: 700;
      letter-spacing: -0.005em; color: var(--fg);
      margin: 1.1em 0 0.45em; line-height: 1.25;
    }
    .markdown-body h1 { font-size: 1.15em; }
    .markdown-body h2 { font-size: 1.08em; }
    .markdown-body h3 { font-size: 1.02em; }
    .markdown-body h4 { font-size: 0.98em; }
    .markdown-body ul, .markdown-body ol { margin: 0.45em 0 0.7em 1.4em; }
    .markdown-body li { margin: 0.2em 0; line-height: 1.5; }
    .markdown-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.86em; background: rgba(0,0,0,0.06);
      padding: 0.06em 0.32em; border-radius: 4px;
    }
    .markdown-body pre {
      background: rgba(0,0,0,0.05);
      padding: 0.7rem 0.9rem;
      border-radius: 6px;
      overflow-x: auto;
      margin: 0.6rem 0;
      font-size: 12.5px; line-height: 1.55;
    }
    .markdown-body pre code { background: none; padding: 0; font-size: inherit; }
    .markdown-body blockquote {
      margin: 0.5em 0; padding: 0 0 0 0.9em;
      border-left: 2px solid var(--fg-faint); color: var(--fg-mute);
    }
    .markdown-body strong { font-weight: 700; color: var(--fg); }
    .markdown-body em { font-style: italic; }
    .markdown-body hr { border: 0; border-top: 1px solid var(--grid); margin: 0.9em 0; }
    .markdown-body a { color: var(--accent); border-bottom: 1px dotted var(--fg-faint); text-decoration: none; }
    /* KaTeX should respect the body color */
    .markdown-body .katex { font-size: 1em; color: var(--fg); }
    .markdown-body .katex-display { margin: 0.6em 0; }
    /* Hide pre-render flash: cards are visually inert until JS renders the bodies */
    .markdown-body[data-md] { color: var(--fg-faint); }
    .markdown-body[data-md]::before {
      content: "rendering…";
      font-family: var(--font-sans); font-size: 11px;
      color: var(--fg-faint); letter-spacing: 0.06em;
    }
    .markdown-body[data-md] > * { display: none; }

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

    panel_label = "LoRA r32 on Qwen3-30B-A3B-Base"
    thinking_label = "Qwen3-30B-A3B thinking"

    out = [HTML_TEMPLATE_HEAD]

    out.append("""
    <header class="perspectives-header">
      <a href="../blog_post/diversity.html" class="header-link">Back to writeup</a>
      <a href="https://github.com/scasella/multi-model" class="header-link">GitHub</a>
    </header>

    <main class="main">
      <article class="benchmarks-content">

        <h1>Side-by-side: multi-persona debate vs Qwen3-thinking</h1>
        <p class="subtitle">Twenty reasoning problems, sampled once each from the published multi-persona debate checkpoint and from vanilla Qwen3-30B-A3B-Thinking, rendered next to each other.</p>
        <p class="date">STEPHEN CASELLA · APRIL 2026</p>

        <div class="paper-links">
          <a href="../blog_post/diversity.html" class="btn-pill">Read the writeup</a>
          <a href="https://github.com/scasella/multi-model" class="btn-pill">GitHub repo</a>
        </div>

        <div class="benchmarks-body">

          <div class="tldr">
            <span class="tldr-label">How to read this page</span>
            <p>
              Each row below shows one problem and one fresh sample from each scaffold at <strong>temperature 1.0</strong> &mdash; not best-of-N, not retrieval, just a single shot. The full <code>&lt;mutipersonaDebate&gt;</code> body and the full <code>&lt;think&gt;</code> body are shown alongside each final answer; both are rendered as Markdown with math. The point isn't to crown a winner per row: it's to let the difference in <em>shape</em> of the two reasoning traces speak for itself.
            </p>
            <p>
              The multi-persona debate often arrives at the same answer with a fraction of the tokens; thinking often takes a longer route and lands somewhere different. On the underspecified, paradoxical, and counterfactual problems both arms can fail interestingly &mdash; the brackets matter more than the bottom line.
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
            # For the thinking arm, the reader-facing answer is whatever sits
            # inside the last \boxed{...} in the model's full trace. The full
            # narrative answer remains visible in the reasoning body above.
            # Fall back to a placeholder when the model emits no boxed value
            # (qualitative problems where there is no closed-form answer).
            think_raw = think.get("raw", "") or ""
            boxed = extract_last_boxed(think_raw)
            if boxed is not None and boxed.strip():
                # Wrap in display math so KaTeX typesets it the same way the
                # model intended (numeric or LaTeX expressions both render).
                think_answer = f"$$\\boxed{{{boxed.strip()}}}$$"
            else:
                think_answer = ""

            panel_chars = len(panel.get("raw", "") or "")
            think_chars = len(think.get("raw", "") or "")

            problem_html = html.escape(it["problem"])
            tests_html = html.escape(it.get("tests", ""))
            expected_html = html.escape(it.get("expected", ""))

            def md_block(text: str, kind: str, empty_msg: str) -> str:
                if not text.strip():
                    return f'<div class="{kind}-body empty">{empty_msg}</div>'
                # data-md flag tells the runtime JS to parse this body's text as
                # Markdown + math. The text content stays HTML-escaped on the
                # server side so the page is well-formed before JS runs.
                return f'<div class="{kind}-body markdown-body" data-md="1">{html.escape(text)}</div>'

            panel_answer_block = md_block(
                panel_answer, "answer", "(no &lt;answer&gt; tag emitted)"
            )
            think_answer_block = md_block(
                think_answer, "answer", "(no \\boxed{} in trace — see reasoning above)"
            )
            panel_reasoning_block = md_block(
                panel_reasoning, "reasoning", "(no &lt;mutipersonaDebate&gt; body)"
            )
            think_reasoning_block = md_block(
                think_reasoning, "reasoning", "(no &lt;think&gt; body)"
            )

            # When thinking's reasoning is materially longer than the
            # multi-persona reasoning on the same problem, surface the ratio
            # on the section label so the per-rollout token-cost asymmetry
            # is visible without making the reader open a calculator.
            ratio_pill = ""
            if panel_chars > 0:
                ratio = think_chars / panel_chars
                if ratio >= 2.0:
                    ratio_pill = f' <span class="ratio">{ratio:.1f}× the multi-persona side</span>'

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
              <div class="arm-label">arm <span class="pill">mutli-persona</span></div>
              <div class="arm-name">{html.escape(panel_label)}</div>
              <div class="section-label">Reasoning <span style="color:var(--fg-faint);font-weight:400;">· {panel_chars:,} chars</span></div>
              {panel_reasoning_block}
              <div class="section-label">Answer</div>
              {panel_answer_block}
            </div>
            <div class="answer-card thinking">
              <div class="arm-label">arm <span class="pill">thinking</span></div>
              <div class="arm-name">{html.escape(thinking_label)}</div>
              <div class="section-label">Reasoning <span style="color:var(--fg-faint);font-weight:400;">· {think_chars:,} chars</span>{ratio_pill}</div>
              {think_reasoning_block}
              <div class="section-label">Answer</div>
              {think_answer_block}
            </div>
          </div>
        </div>
""")

        out.append("      </div>\n    </section>\n")

    # Footnote with metadata
    out.append(f"""
    <div class="footnote">
      <p>
        Sampled at temperature {meta.get('temperature', 1.0)} with <code>max_tokens={meta.get('max_tokens', '—')}</code>; one sample per problem per arm. The multi-persona debate side hits
        <code>{html.escape(meta.get('panel_checkpoint', ''))}</code>; the thinking side hits the public <code>{html.escape(meta.get('thinking_backbone', ''))}</code> via <code>apply_chat_template(enable_thinking=True)</code>.
        Generated {html.escape(meta.get('produced_at', ''))} from <code>{html.escape(meta.get('panel_backbone', ''))}</code>. Source: <a href="https://github.com/scasella/multi-model/blob/case-study/scripts/build_case_study_transcripts.py">build_case_study_transcripts.py</a>.
      </p>
    </div>
""")

    out.append("""
  </div>

  <!-- Markdown + math rendering. Strategy: parse Markdown for ALL bodies
       eagerly (fast — just string manipulation) so anyone scrolling sees
       headings/lists/code immediately. KaTeX is heavy on rendered DOMs so
       we lazy-render math only when a body comes near the viewport. -->
  <script defer src="https://cdn.jsdelivr.net/npm/marked@12.0.0/marked.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <script defer>
    window.addEventListener('DOMContentLoaded', function () {
      const KATEX_OPTS = {
        // Only unambiguous delimiters: single-$ would catch every "$50" / "$8"
        // in the prose and produce nonsense math.
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '\\\\[', right: '\\\\]', display: true},
          {left: '\\\\(', right: '\\\\)', display: false},
        ],
        throwOnError: false,
        errorColor: '#C18549',
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      };

      // 1) Parse all Markdown bodies up front, in chunks so we don't block.
      let pending = Array.from(document.querySelectorAll('.markdown-body[data-md]'));
      function parseChunk() {
        if (typeof marked === 'undefined') {
          // marked still loading; try again shortly.
          setTimeout(parseChunk, 50);
          return;
        }
        marked.setOptions({ breaks: false, gfm: true, headerIds: false, mangle: false });
        const start = performance.now();
        while (pending.length && (performance.now() - start) < 30) {
          const el = pending.shift();
          if (!el || !el.hasAttribute('data-md')) continue;
          try {
            el.innerHTML = marked.parse(el.textContent);
          } catch (e) {
            // leave the raw text in place if parse fails
          }
          el.removeAttribute('data-md');
          el.dataset.mdReady = '1';
        }
        if (pending.length) setTimeout(parseChunk, 0);
        else setupLazyMath();
      }

      // 2) Math rendering is lazy. Each rendered body becomes a
      //    candidate; an IntersectionObserver renders KaTeX as the
      //    body approaches the viewport.
      function setupLazyMath() {
        if (typeof renderMathInElement === 'undefined') {
          setTimeout(setupLazyMath, 100);
          return;
        }
        const obs = new IntersectionObserver(function (entries) {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const el = entry.target;
            obs.unobserve(el);
            try { renderMathInElement(el, KATEX_OPTS); }
            catch (e) { /* ignore */ }
          }
        }, { rootMargin: '400px 0px' });
        document.querySelectorAll('.markdown-body[data-md-ready]').forEach(el => obs.observe(el));
      }

      parseChunk();
    });
  </script>
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
