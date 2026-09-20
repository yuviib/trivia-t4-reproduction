"""
Expand the training set beyond the 10-example demo file bundled with TRivia.

The original tablevqa_samples.jsonl ships with the ground-truth ("assistant")
field hardcoded to the placeholder "<table></table>" for every row, which
means TEDS (the structural reward) can never score anything above zero
against it. This script pulls a larger slice directly from the upstream
source dataset (terryoo/TableVQA-Bench, fintabnetqa split on Hugging Face),
which has real ground-truth HTML tables, real images, and real QA pairs, and
writes a new tablevqa_samples.jsonl with:
  - the corrected HTML-output prompt (the original prompt asks for OTSL,
    which base Qwen2.5-VL has no exposure to; see reproduction/WRITEUP.md)
  - real ground-truth HTML in both the assistant message and a top-level
    "solution" field (which TEDSRewardFunction reads directly)
  - image paths resolved relative to the repo root, matching how ms-swift
    resolves dataset image paths against the training process's cwd

Usage (from the repo root):
    python reproduction/scripts/build_dataset.py --n 40
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "reproduction", "data")
OUT_JSONL = os.path.join(REPO_ROOT, "training", "exps", "tablevqa_samples.jsonl")

PROMPT_TEXT = (
    "You are an AI specialized in recognizing and extracting tables from images. "
    "Your mission is to analyze the table in the image and reproduce it as a valid "
    "HTML table using <table>, <tr>, <td> (and <th> for header cells) tags. "
    "Output only the HTML table and nothing else.\n<image>"
)

DATASET = "terryoo/TableVQA-Bench"
SPLIT = "fintabnetqa"


def fetch_rows(offset: int, length: int) -> dict:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET.replace('/', '%2F')}&config=default&split={SPLIT}"
        f"&offset={offset}&length={length}"
    )
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def main(n_examples: int, batch: int) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rows_out = []
    offset = 0

    while len(rows_out) < n_examples:
        data = fetch_rows(offset, min(batch, n_examples - len(rows_out)))
        if not data["rows"]:
            break
        for item in data["rows"]:
            row = item["row"]
            idx = item["row_idx"]
            img_path = os.path.join(DATA_DIR, f"tablevqa-fintabnetqa-{idx}.jpg")
            urllib.request.urlretrieve(row["image"]["src"], img_path)

            gt_html = row["text_html_table"]
            rel_img_path = os.path.relpath(img_path, REPO_ROOT).replace(os.sep, "/")
            rows_out.append({
                "messages": [
                    {"role": "user", "content": PROMPT_TEXT},
                    {"role": "assistant", "content": gt_html},
                ],
                "qa_pairs": [{"question": row["question"], "answer": row["gt"]}],
                "solution": gt_html,
                "images": [rel_img_path],
            })
        offset += batch
        print(f"fetched {len(rows_out)} / {n_examples}")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows_out)} rows to {OUT_JSONL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="number of examples to pull")
    parser.add_argument("--batch", type=int, default=25, help="rows per API request")
    args = parser.parse_args()
    main(args.n, args.batch)
