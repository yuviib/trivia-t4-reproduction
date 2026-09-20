"""
Call the QA judge reward function directly on a single (question, completion,
ground-truth) triple, outside of training. This is the tool that diagnosed
why QA_F1 stayed at exactly 0.0 across an entire training run even on
well-formed completions: it isolates whether the failure is an exception
being silently swallowed, a genuinely wrong judge answer, or an
answer-matching bug, by printing the raw judge response before any scoring
happens.

Usage (from the repo root):
    OPENAI_API_KEY=sk-... python reproduction/scripts/test_reward_function.py \\
        --question "What was the volatility in 2015?" \\
        --answer "39%" \\
        --html-file path/to/completion.html
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QA_REWARD_MODEL", "gpt-4o-mini")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MS_SWIFT_PATH = os.path.join(REPO_ROOT, "training", "ms-swift")
if MS_SWIFT_PATH not in sys.path:
    sys.path.insert(0, MS_SWIFT_PATH)
REWARD_PLUGIN_PATH = os.path.join(REPO_ROOT, "training", "exps", "trivia_reward_plugin.py")


def load_reward_plugin():
    spec = importlib.util.spec_from_file_location("trivia_reward_plugin", REWARD_PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(question: str, gt_answer: str, html: str) -> None:
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        raise RuntimeError("OPENAI_API_KEY must be set.")

    plugin = load_reward_plugin()
    scorer = plugin.OpenAIQAF1Score()

    raw_answer = scorer._answer_question(question, html)
    print("RAW JUDGE RESPONSE:", repr(raw_answer))

    student = plugin._extract_answer(raw_answer)
    print("extracted student answer:", repr(student))

    gt = plugin._extract_answer(gt_answer)
    print("extracted ground truth:", repr(gt))

    score = plugin._f1_score(student, gt)
    print("f1 score:", score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer", required=True, help="ground-truth answer text")
    parser.add_argument("--html-file", required=True, help="path to a file containing the model's HTML completion")
    args = parser.parse_args()
    with open(args.html_file, encoding="utf-8") as f:
        html = f.read()
    main(args.question, args.answer, html)
