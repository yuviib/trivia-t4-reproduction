"""
Compare the base model against a trained LoRA checkpoint across the full
training set, all three reward functions (QA_F1, TEDS, structural), N
completions per row. This is the script that produced the final before/after
numbers in reproduction/WRITEUP.md.

Uses ms-swift's own PtEngine, the same inference code path GRPOTrainer uses
internally, rather than hand-rolled transformers calls, since that's the
code path already proven to produce real, non-degenerate completions.

Usage (from the repo root, with the training environment set up):
    OPENAI_API_KEY=sk-... python reproduction/scripts/before_after_eval.py \\
        --adapter /path/to/checkpoint-80 \\
        --n-samples 2

The original sweep used temperature 1.0 and 2 samples per row, so each per-row
score is the mean of two noisy samples through an LLM judge. To check whether
a delta is real rather than sampling noise, re-run with a fixed seed and more
samples, or with greedy decoding:
    --seed 0 --n-samples 8
    --temperature 0                (greedy; forces --n-samples 1)
Pass --out results.json to save per-row scores. The summary also prints paired
statistics over rows (win/loss/tie counts, sign test, bootstrap CI).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QA_REWARD_MODEL", "gpt-4o-mini")
os.environ.setdefault("MAX_PIXELS", str(512 * 28 * 28))
os.environ.setdefault("MIN_PIXELS", str(256 * 28 * 28))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS_SWIFT_PATH = os.path.join(REPO_ROOT, "training", "ms-swift")
if MS_SWIFT_PATH not in sys.path:
    sys.path.insert(0, MS_SWIFT_PATH)

DATASET_PATH = os.path.join(REPO_ROOT, "training", "exps", "tablevqa_samples.jsonl")
REWARD_PLUGIN_PATH = os.path.join(REPO_ROOT, "training", "exps", "trivia_reward_plugin.py")

BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT_TEXT = (
    "You are an AI specialized in recognizing and extracting tables from images. "
    "Your mission is to analyze the table in the image and reproduce it as a valid "
    "HTML table using <table>, <tr>, <td> (and <th> for header cells) tags. "
    "Output only the HTML table and nothing else.\n<image>"
)


def load_reward_plugin():
    spec = importlib.util.spec_from_file_location("trivia_reward_plugin", REWARD_PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired_stats(base_rows, trained_rows, n_boot=10000, seed=0):
    """Paired comparison over per-row mean scores (rows are the unit of analysis)."""
    deltas = [t - b for b, t in zip(base_rows, trained_rows)]
    up = sum(d > 0.01 for d in deltas)
    down = sum(d < -0.01 for d in deltas)
    tie = len(deltas) - up - down
    n = up + down
    # two-sided exact sign test on the non-tied rows
    k = min(up, down)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(n_boot)
    )
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]
    return {"up": up, "down": down, "tie": tie, "sign_test_p": p,
            "mean_delta": sum(deltas) / len(deltas), "boot95": [lo, hi]}


def main(adapter_path: str | None, n_samples: int, max_new_tokens: int,
         temperature: float, seed: int | None, out_path: str | None) -> None:
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise RuntimeError("OPENAI_API_KEY must be set (used as the QA judge model).")

    import torch
    from swift.llm import PtEngine, InferRequest, RequestConfig

    with open(DATASET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"{len(rows)} dataset rows")

    engine = PtEngine(BASE_MODEL, torch.float16, use_hf=True, attn_impl="sdpa", max_batch_size=0)

    plugin = load_reward_plugin()
    qa_scorer = plugin.OpenAIQAF1Score()
    teds_scorer = plugin.TEDSRewardFunction()
    struct_scorer = plugin.TableStructureReward()

    if temperature == 0 and n_samples != 1:
        print("temperature 0 is deterministic; forcing --n-samples 1")
        n_samples = 1

    request_config = RequestConfig(
        max_tokens=max_new_tokens, temperature=temperature, top_k=50,
        repetition_penalty=1.15, n=n_samples,
    )
    if seed is not None and hasattr(request_config, "seed"):
        request_config.seed = seed

    def run_pass(label: str):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        qa_all, teds_all, struct_all = [], [], []
        qa_rows = []
        dataset_dir = os.path.dirname(DATASET_PATH)
        for i, row in enumerate(rows):
            image_path = os.path.join(REPO_ROOT, row["images"][0])
            qa_pairs = row["qa_pairs"]
            solution = row.get("solution", row["messages"][1]["content"])
            req = InferRequest(messages=[{"role": "user", "content": PROMPT_TEXT}], images=[image_path])
            resp = engine.infer([req], request_config)
            completions = [c.message.content.strip() for c in resp[0].choices]

            qa_r = qa_scorer(completions=completions, qa_pairs=[qa_pairs] * len(completions))
            teds_r = teds_scorer(completions=completions, solution=[solution] * len(completions))
            struct_r = struct_scorer(completions=completions)

            qa_rows.append(sum(qa_r) / len(qa_r))
            qa_all.extend(qa_r)
            teds_all.extend(teds_r)
            struct_all.extend(struct_r)

            print(f"[{label}] row {i}: QA={sum(qa_r)/len(qa_r):.2f}  "
                  f"TEDS={sum(teds_r)/len(teds_r):.2f}  struct={sum(struct_r)/len(struct_r):.2f}")
            torch.cuda.empty_cache()
        return qa_all, teds_all, struct_all, qa_rows

    print("\n=== BASE PASS ===")
    base_qa, base_teds, base_struct, base_rows = run_pass("BASE")

    if adapter_path:
        print(f"\nAttaching adapter: {adapter_path}")
        engine._add_adapter(adapter_path)
        print("\n=== TRAINED PASS ===")
        trained_qa, trained_teds, trained_struct, trained_rows = run_pass("TRAINED")

        def mean(x):
            return sum(x) / len(x)

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"QA_F1     : base={mean(base_qa):.4f}  trained={mean(trained_qa):.4f}  "
              f"delta={mean(trained_qa) - mean(base_qa):+.4f}")
        print(f"TEDS      : base={mean(base_teds):.4f}  trained={mean(trained_teds):.4f}  "
              f"delta={mean(trained_teds) - mean(base_teds):+.4f}")
        print(f"Structure : base={mean(base_struct):.4f}  trained={mean(trained_struct):.4f}  "
              f"delta={mean(trained_struct) - mean(base_struct):+.4f}")

        stats = paired_stats(base_rows, trained_rows)
        print(f"\nQA_F1 paired over {len(base_rows)} rows "
              f"(temperature={temperature}, n_samples={n_samples}, seed={seed}):")
        print(f"  rows up/down/tied : {stats['up']}/{stats['down']}/{stats['tie']}")
        print(f"  sign test p       : {stats['sign_test_p']:.3f}")
        print(f"  mean delta        : {stats['mean_delta']:+.4f}  "
              f"bootstrap 95% CI [{stats['boot95'][0]:+.4f}, {stats['boot95'][1]:+.4f}]")

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "adapter": adapter_path, "temperature": temperature,
                    "n_samples": n_samples, "seed": seed,
                    "base_qa_rows": base_rows, "trained_qa_rows": trained_rows,
                    "paired_qa": stats,
                }, f, indent=2)
            print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None, help="path to a LoRA checkpoint to compare against base")
    parser.add_argument("--n-samples", type=int, default=2, help="completions sampled per row per model")
    parser.add_argument("--max-new-tokens", type=int, default=576)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature; 0 = greedy (the original sweep used 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="fix the sampling seed for both passes")
    parser.add_argument("--out", type=str, default=None,
                        help="write per-row scores and paired stats to this JSON file")
    args = parser.parse_args()
    main(args.adapter, args.n_samples, args.max_new_tokens, args.temperature, args.seed, args.out)
