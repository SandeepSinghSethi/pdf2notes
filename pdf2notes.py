#!/usr/bin/env python3
"""
pdf2notes — Turn a page range of a PDF book into descriptive, review-ready
study notes (Markdown), ready to drop into Obsidian or Notion.

Works with ANY OpenAI-compatible chat completion API: NVIDIA NIM
(integrate.api.nvidia.com), OpenAI, Groq, Together AI, OpenRouter, a local
vLLM/Ollama server, etc. You bring your own API key and base URL.

Requests run concurrently (bounded by --concurrency) and are throttled to
stay under --rpm (requests/minute), so large page ranges finish in minutes
instead of hours without tripping your provider's rate limit.

USAGE
-----
    python pdf2notes.py --pdf book.pdf --pages 120-180 \
        --api-key-env NVIDIA_API_KEY \
        --base-url https://integrate.api.nvidia.com/v1 \
        --model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
        --style descriptive \
        --concurrency 5 --rpm 40 \
        --output notes.md

Set the API key as an environment variable first, e.g.:
    export NVIDIA_API_KEY="nvapi-xxxxxxxx"

NOTE: NVIDIA (and other providers) periodically retire model IDs. If you
get an HTTP 404/410 error, check the current catalog at
https://build.nvidia.com/models and pass the correct --model.

If a run is interrupted or fails partway, just rerun the exact same
command — chunks already written to the output file are skipped.

Run `python pdf2notes.py --help` for all options.
"""

import argparse
import asyncio
import os
import re
import sys
import time
from collections import deque
from datetime import datetime

import pdfplumber
from openai import (
    AsyncOpenAI,
    APIStatusError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


# ---------------------------------------------------------------------------
# Prompt templates per note style
# ---------------------------------------------------------------------------

STYLE_PROMPTS = {
    "descriptive": (
        "You are an expert study-notes writer. Turn the given book excerpt into "
        "detailed, descriptive notes a learner can review later WITHOUT rereading "
        "the original text. Explain concepts in your own words, don't just "
        "shorten sentences. Include definitions, examples, cause-effect "
        "relationships, and any numbers, names, or terms that matter. "
        "Use Markdown headers (##, ###) for structure and bullet points for "
        "detail. Where useful, add a one-line 'Why it matters' or 'Key "
        "takeaway' callout. Do not skip content just because it seems minor — "
        "the goal is a faithful, learnable rewrite, not a compressed summary."
    ),
    "cornell": (
        "You are an expert study-notes writer using the Cornell Notes method. "
        "For the given book excerpt, produce Markdown with three parts: "
        "1) '## Cues' — a bullet list of key terms/questions, "
        "2) '## Notes' — detailed explanatory notes matching each cue, "
        "3) '## Summary' — a short paragraph summarizing the excerpt. "
        "Be thorough in the Notes section — this is what the learner studies from."
    ),
    "qa": (
        "You are an expert study-notes writer. Convert the given book excerpt "
        "into a thorough set of Markdown Q&A flashcard-style notes: "
        "'### Q: <question>' followed by a detailed '**A:** <answer>'. Cover "
        "every important concept, definition, and detail in the excerpt — "
        "write enough questions to fully capture the material, not just the "
        "obvious ones."
    ),
    "outline": (
        "You are an expert study-notes writer. Convert the given book excerpt "
        "into a deeply nested Markdown outline (using #, ##, ###, and nested "
        "bullet points) that mirrors the structure of the content and captures "
        "every important detail, definition, and example in the source."
    ),
}

SYSTEM_SUFFIX = (
    "\n\nFormatting rules:\n"
    "- Output ONLY Markdown, no preamble like 'Here are the notes'.\n"
    "- Start directly with a level-2 heading (##) that names the topic of "
    "this excerpt.\n"
    "- Reference page numbers in parentheses, e.g. (p. 42), when introducing "
    "a major concept, so the notes stay traceable to the source.\n"
    "- Never say things like 'the author discusses' — just present the "
    "content directly, as notes.\n"
)


# ---------------------------------------------------------------------------
# PDF extraction (memory-conscious: flush pdfplumber's per-page cache
# immediately after pulling text, instead of letting fonts/layout objects
# for the whole range pile up in memory)
# ---------------------------------------------------------------------------

def extract_pages(pdf_path, start_page, end_page):
    """Returns list of (page_number, text) for the given 1-indexed inclusive range."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        if start_page < 1 or end_page > total or start_page > end_page:
            raise ValueError(
                f"Invalid page range {start_page}-{end_page}. "
                f"This PDF has {total} pages."
            )
        for i in range(start_page - 1, end_page):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            pages.append((i + 1, text))
            # Release this page's cached fonts/chars/layout objects now —
            # otherwise pdfplumber keeps them alive for the whole range.
            page.flush_cache()
    return pages


def chunk_pages(pages, max_chars=6000):
    """
    Groups consecutive pages into chunks under max_chars, so each API call
    covers a coherent span of pages. Returns list of (first_pg, last_pg, text).
    """
    chunks = []
    cur_text = []
    cur_start = None
    cur_len = 0
    last_pg_seen = None

    for pg_num, text in pages:
        last_pg_seen = pg_num
        text = text.strip()
        if not text:
            continue
        if cur_start is None:
            cur_start = pg_num
        addition = f"\n\n[--- page {pg_num} ---]\n{text}"
        if cur_len + len(addition) > max_chars and cur_text:
            chunks.append((cur_start, pg_num - 1, "".join(cur_text)))
            cur_text = [addition]
            cur_start = pg_num
            cur_len = len(addition)
        else:
            cur_text.append(addition)
            cur_len += len(addition)

    if cur_text:
        chunks.append((cur_start, last_pg_seen, "".join(cur_text)))

    return chunks


# ---------------------------------------------------------------------------
# Resume support: figure out which chunks are already in the output file
# ---------------------------------------------------------------------------

def load_completed_ranges(output_path):
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        (int(a), int(b))
        for a, b in re.findall(r"<!-- pages (\d+)-(\d+) -->", content)
    }


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, requests/minute) — shared across all
# concurrent workers so total request rate stays under your provider's limit
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rpm):
        self.rpm = max(rpm, 1)
        self.lock = asyncio.Lock()
        self.timestamps = deque()

    async def acquire(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] > 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.rpm:
                    self.timestamps.append(now)
                    return
                wait = 60 - (now - self.timestamps[0]) + 0.05
                await asyncio.sleep(max(wait, 0.05))


# ---------------------------------------------------------------------------
# AI call (async, with retry only for transient errors)
# ---------------------------------------------------------------------------

NON_RETRYABLE_HINT = (
    "This usually means the --model name is wrong or has been retired, or "
    "the API key / --base-url is invalid. Check your provider's current "
    "model list (e.g. https://build.nvidia.com/models) and retry with the "
    "correct --model."
)


async def generate_notes_for_chunk(client, limiter, semaphore, model, style,
                                    chunk_text, first_pg, last_pg, stop_event,
                                    max_retries=4):
    system_prompt = STYLE_PROMPTS[style] + SYSTEM_SUFFIX
    user_prompt = f"Book excerpt covering pages {first_pg}-{last_pg}:\n\n{chunk_text}"

    async with semaphore:
        delay = 2
        for attempt in range(1, max_retries + 1):
            if stop_event.is_set():
                raise RuntimeError("cancelled: a fatal error occurred elsewhere")

            await limiter.acquire()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()

            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"gave up after {max_retries} attempts (pages "
                        f"{first_pg}-{last_pg}): {e}"
                    ) from e
                print(f"  [pages {first_pg}-{last_pg}] transient error "
                      f"({type(e).__name__}), retrying in {delay}s...", file=sys.stderr)
                await asyncio.sleep(delay)
                delay *= 2

            except APIStatusError as e:
                # Non-retryable: bad model, auth, deprecated model (404/410), etc.
                # Retrying this would just waste your rate-limit budget.
                raise RuntimeError(
                    f"HTTP {e.status_code} on pages {first_pg}-{last_pg}: "
                    f"{e.message}. {NON_RETRYABLE_HINT}"
                ) from e


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(args, chunks, output_path, frontmatter, api_key):
    completed = load_completed_ranges(output_path)

    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(frontmatter)

    pending = [
        (i, f, l, t) for i, (f, l, t) in enumerate(chunks) if (f, l) not in completed
    ]

    already_done = len(chunks) - len(pending)
    if already_done:
        print(f"{already_done} chunk(s) already in {output_path} — skipping those.")
    if not pending:
        print("Nothing left to generate.")
        return 0, None

    print(f"Generating {len(pending)} chunk(s) "
          f"(concurrency={args.concurrency}, rate limit={args.rpm}/min)...")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    limiter = RateLimiter(args.rpm)
    semaphore = asyncio.Semaphore(args.concurrency)
    stop_event = asyncio.Event()

    results = {}
    errors = {}

    async def worker(idx, first_pg, last_pg, text):
        if stop_event.is_set():
            return
        try:
            notes = await generate_notes_for_chunk(
                client, limiter, semaphore, args.model, args.style,
                text, first_pg, last_pg, stop_event,
            )
            results[idx] = (first_pg, last_pg, notes)
            print(f"  done: pages {first_pg}-{last_pg} "
                  f"({len(results)}/{len(pending)})")
        except RuntimeError as e:
            errors[idx] = (first_pg, last_pg, e)
            stop_event.set()

    try:
        tasks = [asyncio.create_task(worker(i, f, l, t)) for i, f, l, t in pending]
        await asyncio.gather(*tasks)
    finally:
        await client.close()

    # Flush a *contiguous* prefix of newly-completed chunks in page order.
    # Anything after the first gap (still running/failed) is left for a
    # rerun — resume picks it up via the <!-- pages a-b --> markers.
    written = 0
    with open(output_path, "a", encoding="utf-8") as fh:
        for i, (f, l, _t) in enumerate(chunks):
            if (f, l) in completed:
                continue
            if i not in results:
                break
            first_pg, last_pg, notes = results[i]
            fh.write(f"\n\n<!-- pages {first_pg}-{last_pg} -->\n{notes}\n")
            written += 1

    fatal = None
    if errors:
        first_idx = min(errors)
        fatal = errors[first_idx]

    return written, fatal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_page_range(s):
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError("Pages must look like START-END, e.g. 120-180")
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF page range into AI-generated study notes."
    )
    parser.add_argument("--pdf", required=True, help="Path to the source PDF")
    parser.add_argument("--pages", required=True, type=parse_page_range,
                         help="Page range, e.g. 120-180 (inclusive, 1-indexed)")
    parser.add_argument("--output", default=None,
                         help="Output .md file (default: <pdf-name>_notes_<range>.md)")
    parser.add_argument("--style", choices=list(STYLE_PROMPTS.keys()),
                         default="descriptive", help="Note style (default: descriptive)")
    parser.add_argument("--chunk-chars", type=int, default=6000,
                         help="Max characters of source text per API call (default 6000)")
    parser.add_argument("--concurrency", type=int, default=5,
                         help="Max simultaneous API requests (default 5)")
    parser.add_argument("--rpm", type=int, default=40,
                         help="Max API requests per minute, across all workers (default 40) "
                              "— set this to match your provider's actual rate limit")
    parser.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b",
                         help="Model name as your provider expects it. Providers retire "
                              "model IDs periodically — check your provider's current "
                              "catalog if this default 404s/410s.")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1",
                         help="OpenAI-compatible API base URL "
                              "(default: NVIDIA NIM). Use https://api.openai.com/v1 "
                              "for OpenAI, http://localhost:11434/v1 for local, etc.")
    parser.add_argument("--api-key", default=None,
                         help="API key (avoid this on shared machines — prefer --api-key-env)")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY",
                         help="Env var name holding the API key (default: NVIDIA_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"No API key found. Set it with:\n"
            f"  export {args.api_key_env}=\"your-key-here\"\n"
            f"or pass --api-key directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    start_page, end_page = args.pages
    output_path = args.output or (
        f"{os.path.splitext(os.path.basename(args.pdf))[0]}"
        f"_notes_{start_page}-{end_page}.md"
    )

    print(f"Extracting pages {start_page}-{end_page} from {args.pdf} ...")
    pages = extract_pages(args.pdf, start_page, end_page)

    non_empty = sum(1 for _, t in pages if t.strip())
    if non_empty == 0:
        print(
            "No extractable text found in this range — the PDF might be "
            "scanned/image-based. This tool needs OCR'd text first "
            "(e.g. run it through an OCR tool, then retry).",
            file=sys.stderr,
        )
        sys.exit(1)

    chunks = chunk_pages(pages, max_chars=args.chunk_chars)
    del pages  # done with raw page text; chunks hold what we still need
    print(f"Split into {len(chunks)} chunk(s) for note generation.")

    book_title = os.path.splitext(os.path.basename(args.pdf))[0].replace("_", " ")
    frontmatter = (
        "---\n"
        f"title: \"{book_title} — Notes (pp. {start_page}-{end_page})\"\n"
        f"source: \"{os.path.basename(args.pdf)}\"\n"
        f"pages: \"{start_page}-{end_page}\"\n"
        f"style: {args.style}\n"
        f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "tags: [book-notes]\n"
        "---\n\n"
    )

    written, fatal = asyncio.run(
        run_pipeline(args, chunks, output_path, frontmatter, api_key)
    )

    if fatal:
        first_pg, last_pg, err = fatal
        print(f"\nStopped due to an error on pages {first_pg}-{last_pg}:\n  {err}",
              file=sys.stderr)
        if written:
            print(f"{written} new chunk(s) were still saved to {output_path}.",
                  file=sys.stderr)
        print("Fix the issue above, then rerun the exact same command — "
              "completed chunks are skipped automatically.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Notes saved to: {output_path}")
    print("Drop this file straight into your Obsidian vault, or import into Notion "
          "(Import > Markdown).")


if __name__ == "__main__":
    main()
