from __future__ import annotations

import concurrent.futures
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List

import jieba
from openai import OpenAI
from swift.plugin import ORM, orms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils.teds import normalize_table_text, teds_score


EN_LLM_PROMPT = """Given an HTML-formatted table and a corresponding question, your task is to respond appropriately based on table. If the table do not contain the answer of question, output "Not answerable".
Your answer should be a short phrase of only few words. Output the answer within <answer> </answer>.

HTML Table: {}

Question: {}"""

ZH_LLM_PROMPT = """给定一个HTML格式的表格以及一个相应的问题，你的任务是根据表格回答该问题。如果该表格不包含该问题的答案，请输出"无法回答"。你的答案必须简短、仅有一两个词语。输出答案时用<answer></answer>包裹。

HTML表格: {}

问题: {}"""


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _extract_answer(text: str | None) -> str:
    if text is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text)
    return match.group(1).strip() if match else text.strip()


def _f1_score(prediction: str, ground_truth: str) -> float:
    if prediction.startswith("ERROR"):
        return 0.0
    if "not answerable" in prediction.lower():
        return 0.0

    prediction_tokens = " ".join(jieba.cut(prediction))
    ground_truth_tokens = " ".join(jieba.cut(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


class OpenAIQAF1Score(ORM):
    def __init__(self, args=None, **kwargs) -> None:
        super().__init__()
        self.clients = [OpenAI(api_key=os.environ["OPENAI_API_KEY"])]

    def _answer_question(self, question: str, completion: str) -> str:
        try:
            html_table = normalize_table_text(completion)
            if len(html_table) > 30000:
                html_table = html_table[:30000]

            prompt = ZH_LLM_PROMPT if _contains_chinese(question) else EN_LLM_PROMPT
            response = random.choice(self.clients).chat.completions.create(
                model=os.environ.get("QA_REWARD_MODEL", "Qwen/Qwen3.5-9B"),
                messages=[{"role": "user", "content": prompt.format(html_table, question)}],
                max_tokens=100,
                temperature=0,
                top_p=0.8,
                presence_penalty=1.5,
            )
            return response.choices[0].message.content
        except Exception:
            return "ERROR"

    def _batch_answer(self, questions: list[str], completions: list[str]) -> list[str]:
        max_workers = min(int(os.environ.get("QA_REWARD_WORKERS", "120")), len(completions))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda item: self._answer_question(*item), zip(questions, completions)))

    def __call__(self, completions, qa_pairs=None, **kwargs):
        if qa_pairs is None:
            return [0.0] * len(completions)

        questions = []
        completion_inputs = []
        completion_indices = []
        for completion_index, (completion, completion_qa_pairs) in enumerate(zip(completions, qa_pairs)):
            for qa_pair in completion_qa_pairs:
                questions.append(qa_pair["question"])
                completion_inputs.append(completion)
                completion_indices.append(completion_index)

        answers = self._batch_answer(questions, completion_inputs)
        completion_rewards = [[] for _ in completions]

        qa_offsets = [0]
        for completion_qa_pairs in qa_pairs:
            qa_offsets.append(qa_offsets[-1] + len(completion_qa_pairs))

        for qa_index, answer in enumerate(answers):
            try:
                completion_index = completion_indices[qa_index]
                qa_pair = qa_pairs[completion_index][qa_index - qa_offsets[completion_index]]
                gt_answer = _extract_answer(qa_pair["answer"])
                student_answer = _extract_answer(answer)
                reward = _f1_score(student_answer, gt_answer)
            except Exception:
                reward = 0.0

            completion_rewards[completion_index].append(reward)

        return [
            sum(item_rewards) / len(item_rewards) if item_rewards else 0.0
            for item_rewards in completion_rewards
        ]


class TEDSRewardFunction(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, reference in zip(completions, solution):
            try:
                prediction = _extract_answer(completion)
                ground_truth = _extract_answer(reference)

                if len(prediction) > 30000:
                    prediction = prediction[:30000]
                if len(ground_truth) > 30000:
                    ground_truth = ground_truth[:30000]

                if ground_truth == "<table></table>":
                    rewards.append(0.0)
                    continue

                rewards.append(
                    teds_score(
                        normalize_table_text(prediction),
                        normalize_table_text(ground_truth),
                    )
                )
            except Exception:
                rewards.append(0.0)
        return rewards


class TableStructureReward(ORM):
    """Self-supervised structural validity reward. No ground truth required.
    Gives partial credit for well-formed table HTML so the model has *some*
    gradient signal even when QA_F1 scores 0, which is the common case with
    a small, sparse-reward dataset.
    """

    def __call__(self, completions, **kwargs):
        return [self._score(c) for c in completions]

    def _score(self, completion: str) -> float:
        try:
            text = completion.strip()
            text = re.sub(r"^```(?:html)?\s*", "", text)
            text = re.sub(r"```\s*$", "", text).strip()
            lower = text.lower()

            score = 0.0
            if "<table" not in lower:
                return 0.0
            score += 0.2
            if "</table>" not in lower:
                return score
            score += 0.2

            row_tags = re.findall(r"<tr[\s>]", lower)
            if len(row_tags) < 2:
                return score
            score += 0.2

            rows = re.split(r"<tr[^>]*>", text, flags=re.IGNORECASE)[1:]
            cell_counts = []
            for row in rows:
                row = row.split("</tr>")[0]
                cells = re.findall(r"<t[dh][\s>]", row, flags=re.IGNORECASE)
                if cells:
                    cell_counts.append(len(cells))
            if not cell_counts:
                return score
            score += 0.2

            most_common_len, freq = Counter(cell_counts).most_common(1)[0]
            score += 0.2 * (freq / len(cell_counts))

            junk_markers = [
                "<script", "<style", "<button", "click me",
                "i'm sorry", "as an ai", "this code snippet",
            ]
            if any(m in lower for m in junk_markers):
                score *= 0.5

            return min(score, 1.0)
        except Exception:
            return 0.0


orms["QA_F1_score"] = OpenAIQAF1Score
orms["TEDS"] = TEDSRewardFunction
orms["TableStructure"] = TableStructureReward
