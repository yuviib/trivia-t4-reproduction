# Reproducing and extending TRivia: self-supervised GRPO fine-tuning for table recognition on a single free GPU

**Base paper:** TRivia (CVPR 2026), [arXiv:2512.01248](https://arxiv.org/abs/2512.01248), code and model at [HKU-TASR/TRivia](https://github.com/HKU-TASR/TRivia)
**Model:** Qwen2.5-VL-3B-Instruct
**Compute:** a single free-tier Kaggle T4 GPU (fp16, no bf16 support)

## TL;DR

TRivia fine-tunes a vision-language model to extract tables from images without any labeled ground-truth tables, using GRPO and a question-answering-based reward: an LLM judge tries to answer questions about the extracted table, and how well it can answer becomes the training signal. The paper trains full-parameter, on 8 GPUs, with an automatic question-generation module and a sample-selection mechanism over a large unlabeled corpus.

This project reproduces the core RL optimization loop at roughly one-hundredth the scale: LoRA instead of full fine-tuning, tens of examples instead of a large corpus, a single T4 instead of 8 GPUs, and real questions sourced from an existing benchmark rather than the paper's own auto-generation module. Getting even that much running required finding and fixing four bugs in the released code and abandoning one design (a vLLM rollout server whose weight sync I could not get working). Two extensions were built on top: a working structural (TEDS) reward, which the released demo data cannot support, and a new self-supervised structural reward that is not in the paper.

**What the results show, and don't.** The reward pipeline works end to end: with truncation fixed, QA_F1 was non-zero on 51 of 80 training steps. But the reward curve shows no clear upward trend within the run, and on the 40 training examples the LoRA checkpoint scored lower than base on QA_F1 (0.196 to 0.107; 4 examples up, 11 down, 25 unchanged). That evaluation used 2 samples per example at temperature 1.0 through an LLM judge, and the difference is not statistically distinguishable from noise (sign test p = 0.12). So this reproduction shows that the loop runs, and does not show that the model learned anything at this scale.

## What the paper does

Table recognition, turning a table image into structured HTML, has depended on supervised fine-tuning, which needs labeled ground-truth tables that are expensive to produce. That's locked open-source models out of the top tier, since only well-funded labs can afford enough labeled data to compete.

TRivia removes the labeling requirement with a closed self-supervised loop. An attention-guided module automatically generates diverse questions about a table image. The vision-language model extracts the table. A judge model tries to answer the generated questions using only that extraction as context, right answers mean an accurate extraction, wrong or unanswerable means it wasn't. That correctness, an F1 score over the generated questions, is the reward GRPO uses to update the model. A sample-selection mechanism additionally identifies which unlabeled images are actually useful for learning, rather than training uniformly over everything. The result, TRivia-3B, is reported to beat Gemini 2.5 Pro and MinerU2.5 on three public benchmarks.

## Scope of this reproduction

This project tested the RL optimization loop itself, given real questions, not the full three-part pipeline. The automatic question-generation module and the automatic sample-selection mechanism were not reproduced; questions and images came from an existing public benchmark (TableVQA-Bench, fintabnetqa split) instead. That's a meaningfully smaller claim than "reproduced TRivia," and it's stated plainly here rather than implied otherwise.

| Component | Paper | This reproduction |
|---|---|---|
| Fine-tuning | full-parameter | LoRA, rank 8, alpha 32, language-model layers only, vision tower and aligner frozen (0.4% of parameters trainable) |
| GPUs | 8 | 1 (T4, fp16) |
| Question source | attention-guided auto-generation | existing benchmark (TableVQA-Bench) |
| Sample selection | automatic, over unlabeled corpus | fixed 10 then 40 example set |
| Rollout engine | vLLM | plain HF `generate()` (the vLLM rollout-server weight sync never worked; see [The vLLM attempt](#the-vllm-attempt-not-resolved)) |
| Reward | QA_F1 + TEDS (paper's own default weights TEDS to 0) | QA_F1 + TEDS (made functional, see below) + a new structural reward |

## Bugs found and fixed in the released code

**The demo dataset's prompt asks for OTSL, a format the base model has never seen.** OTSL is a specialized table markup TRivia's authors use internally. Base Qwen2.5-VL-3B has zero exposure to it and produced nothing the reward pipeline could parse, so every reward came back as exactly zero regardless of extraction quality. Rewriting the prompt to ask for plain HTML, something the base model already knows from pretraining, fixed this immediately and produced real, non-zero reward on the very next run.

**fp16 sampling crashed unpredictably on T4 hardware.** T4s don't support bf16, so generation ran in fp16, and partway through nearly every training attempt it crashed with `probability tensor contains either inf, nan or element < 0` inside PyTorch's sampling step. Lowering temperature, adding repetition penalty, and shortening the completion length all failed to reliably fix it, the crash just moved to a different step. The actual fix was a custom `LogitsProcessor` injected into ms-swift's `PtEngine`, sanitizing NaN/inf logits with `torch.nan_to_num()` before the softmax and multinomial sampling step. That eliminated the crash entirely and let every subsequent full run complete cleanly. See `training/ms-swift/swift/llm/infer/infer_engine/pt_engine.py`.

**The structural reward (TEDS) is wired to a ground-truth field that's hardcoded to a placeholder.** The bundled demo dataset sets the ground-truth ("assistant") field to the literal string `"<table></table>"` for every row, and `TEDSRewardFunction` explicitly returns 0.0 whenever it sees that exact placeholder. TEDS could never score anything above zero, not because it was weighted to 0 (the paper's own `run.sh` also weights it to 0 by default), but because it structurally could not function at all with the released demo data. Tracing this back to the real upstream dataset (`terryoo/TableVQA-Bench` on Hugging Face, which has genuine `text_html_table` ground truth) and building a pipeline to pull real annotations made TEDS functional for the first time in this reproduction. See `reproduction/scripts/build_dataset.py`.

**Completions weren't being logged despite `--log_completions true`.** ms-swift's vendored GRPO trainer deletes trl's built-in completion-logging method (`del HFGRPOTrainer.log`) without providing a replacement, so the flag was silently a no-op. Patched `_generate_and_score_completions` to call the trainer's own already-instantiated `JsonlWriter` directly. See `training/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py`.

## The vLLM attempt (not resolved)

The recipe generates rollouts on a separate vLLM server and syncs updated weights into it. I tried this on Kaggle's 2x T4 (rollout server on one GPU, trainer on the other) and made six changes to that path: a `spawn` start method for the rollout process, the server's `load_format` (`dummy` to `auto`), and four successive changes to which parameter names get streamed to the server (skip `visual.*`, apply the skip on a second code path, match `visual.` as a substring, strip the `language_model.` prefix). The server came up and served requests, but weight sync kept failing with `KeyError: 'language_model.embed_tokens.weight'`, and the trainer hit `DistNetworkError` and then `CUDA error: device-side assert triggered`. I did not find the root cause and moved rollouts into the trainer with HF `generate()`, which removes the sync entirely. The scripts and a fuller account are in [`vllm_attempt/`](./vllm_attempt/README.md). The notebook does not let me say whether the device-side asserts came from this path or from the fp16 NaN problem described above, so I don't claim either.

## What was added beyond reproduction

**A working TEDS structural reward**, described above, restoring a component that's present in the reward architecture but non-functional in the public release.

**`TableStructureReward`**, a new self-supervised reward function that doesn't appear anywhere in the paper or its code. It requires no ground truth and no LLM call: it parses a completion for well-formed table structure (a `<table>` tag, closing tag, at least two rows, consistent cell counts per row) and penalizes off-task junk output (stray script/style tags, chatty prose breaking the expected format), giving partial credit purely from structural validity. It was designed specifically in response to a failure mode found empirically: with QA_F1 reward this sparse, the large majority of completions carried zero gradient signal, and this gives the model something to learn from on nearly all of them. See `training/exps/trivia_reward_plugin.py`.

**A larger, properly-labeled training set.** The bundled 10-example file was expanded to 40 examples pulled directly from the source benchmark, each with a real image, a real question-answer pair, and real ground-truth HTML (`reproduction/scripts/build_dataset.py`).

**Diagnostic tooling that isolates failure causes rather than guessing.** `reproduction/scripts/test_reward_function.py` calls the QA judge directly on a single completion, outside of training, printing the raw judge response before any scoring happens, which is what distinguished a genuine judge misreading from a silent exception or an answer-matching bug (see Results below). `reproduction/scripts/before_after_eval.py` runs a controlled base-vs-trained sweep across the full dataset.

## Results

**Run 1 (10 examples, QA_F1 only, num_generations=2):** confirmed the pipeline works end to end. 4 of 32 logged optimizer steps carried non-zero reward. A before/after sweep across the 10 examples showed gains on a couple of examples and regressions on a couple of others that the base model had already solved. I did not record an aggregate for this sweep, so this is a qualitative description only. Likely cause: with reward this sparse (4 of 91 individual logged completions earning anything), a handful of LoRA updates at a tiny learning rate has almost nothing to learn from. I did not test this.

**Run 2 (40 examples, TEDS + structural reward added, num_generations=4, max_completion_length=320):** confirmed TEDS and the structural reward both work correctly with real data (TEDS non-zero on 100% of logged steps). QA_F1 stayed at exactly 0.0 across all 80 steps and 401 completions. Direct testing (`test_reward_function.py`) traced this to two separate, real causes, not a bug: most completions were truncated before finishing a single row (`clipped_ratio: 1.0` on the majority of steps, starving QA_F1 by construction since an incomplete table usually can't answer the target question), and even complete, well-formed completions sometimes got misread by the judge model on ambiguous column layouts, a genuine and inherent limitation of using a cheap LLM as the reward judge, confirmed by testing the exact same completion and question directly and inspecting the raw judge response.

**Run 3 (same as run 2, `max_completion_length` raised to 576):** fixing the truncation issue is what produced QA_F1 signal: non-zero on 51 of 80 steps (mean 0.374, reaching 1.0 repeatedly), with TEDS and the structural reward also non-zero on every step and 101 of 401 completions scoring 0.9 or higher on QA_F1. Run 2 and Run 3 differ only in this one setting, so the jump from 0 of 80 to 51 of 80 non-zero QA_F1 steps is attributable to it, not to the structural reward.

The QA_F1 curve does not trend upward within the run. Mean QA_F1 over successive 20-step blocks is 0.35, 0.39, 0.27 and 0.49, with 14, 12, 10 and 15 non-zero steps. The blocks are noisy and 0.49 is the highest, but with one run and no repeats I can't call that learning.

**Before/after evaluation (Run 3 checkpoint-80 vs. base).** All 40 examples were also the training examples (held-in), 2 samples each at temperature 1.0, judged by gpt-4o-mini:

| Metric | Base | Trained | Delta |
|---|---|---|---|
| QA_F1 | 0.1964 | 0.1065 | -0.0899 |
| TEDS | 0.1654 | 0.1583 | -0.0071 |
| Structure | 0.5385 | 0.5207 | -0.0178 |

Per example, QA_F1 went up on 4, down on 11 and was unchanged on 25 (21 of 40 examples scored 0.00 for both models; base scored 0.00 on 24 examples and trained on 30). Paired over examples, the two-sided sign test on the 15 non-tied examples gives p = 0.12, and a bootstrap 95% interval for the mean QA_F1 delta is roughly [-0.19, 0.00]. So the point estimate is negative and the data are compatible with anything from a modest decline to no change. TEDS and structure are essentially flat. See `reproduction/dashboard.html` for the reward curve and a scatter of every example's before/after score.

## What this does and doesn't show

**Shows:** the released GRPO recipe can be run end to end on one free T4 with LoRA once the bugs above are fixed, the reward functions (QA_F1, TEDS, structural) behave as intended on real data, and the QA_F1 signal depends strongly on completion length (0 of 80 steps at 320 tokens, 51 of 80 at 576).

**Does not show:** that training improved the model. On its own training examples the checkpoint scored no better than base and probably a little worse. Since these are the training examples, this is not a generalization gap. The model did not measurably improve even on data it was optimized against. I have not tested why. Candidates, none verified:
- Too little optimization: 80 steps of LoRA at a small learning rate over 40 diverse examples, possibly with many groups carrying little reward variance (I did not check the logged reward std).
- Evaluation noise: 2 samples at temperature 1.0, a stochastic LLM judge, and most rows scoring 0.00 leave little resolving power.
- Objective mismatch: the QA judge may reward things that the eval scoring, decoding settings (repetition penalty 1.15, top-k 50) or truncation treat differently from training.

**What I'd do to tell these apart:** re-run the evaluation with a fixed seed and 8+ samples per example, and with greedy decoding (`before_after_eval.py --seed 0 --n-samples 8`, and `--temperature 0`), which needs the checkpoint and a GPU and has not been run. If the deficit persists under greedy decoding, evaluation noise is out. A held-out split and a longer run would be the next steps.

## Resume-ready statements

Each of these is limited to what the record supports:

- Ran a research paper's GRPO fine-tuning recipe (TRivia, Qwen2.5-VL-3B) end to end on a single free 15GB T4 using LoRA (14.97M of 3.77B parameters trainable, 0.4%), in place of the paper's 8-GPU full-parameter setup.
- Found and fixed four bugs in the released code: a demo prompt asking for a markup format the base model has never seen (all rewards silently zero), an fp16 sampling crash on GPUs without bf16 (custom logits sanitizer), a structural reward that could never fire because the demo ground truth was a placeholder, and silently disabled completion logging.
- Traced a zero QA reward to completion truncation rather than model quality (using logged clipped-completion ratios and direct reward-function testing); raising the completion limit took QA_F1 from non-zero on 0 of 80 steps to 51 of 80.
- Added a label-free structural reward (well-formed table, consistent row widths, no off-task text) not present in the paper.
- Evaluated a trained checkpoint against base on all three reward metrics and reported the result as inconclusive: QA_F1 0.196 to 0.107 on the training examples (sign test p = 0.12), with the evaluation's noise sources identified.
- Attempted the paper's vLLM rollout-server design and documented why it was abandoned after six patches and an unresolved weight-sync failure.

## What I'd do next

1. **Resolve the evaluation question first.** Re-run the before/after sweep with a fixed seed and more samples per example, and with greedy decoding, on a held-out split as well as the training examples. This is cheap (inference only) and decides whether there is a real effect to explain.
2. **Then scale the training signal.** `num_generations`, dataset size and question diversity all affect how often a group has reward variance, which is what GRPO learns from. That needs more compute than a free T4 (Modal or similar) and probably the vLLM rollout path.
3. **Test the part of the pipeline I skipped.** Using the paper's own auto-generated questions instead of curated benchmark questions would show whether question quality is another source of reward sparsity.

Testing sensitivity to the paper's own auto-generated questions, rather than the curated benchmark questions used here, would also directly test a part of the pipeline this reproduction skipped: if the question-generation module produces low-quality or repetitive questions on certain table types, the same reward-sparsity failure mode found here would likely reappear in the full pipeline too.

## Reproducing this

```
./setup.sh          # creates ./TRivia: upstream + fixes + these scripts
cd TRivia
pip install -e training/ms-swift
pip install "qwen_vl_utils[decord]<0.0.12" apted math_verify==0.5.2 jieba openai python-Levenshtein

# optional: expand the training set beyond the bundled 10 examples
python reproduction/scripts/build_dataset.py --n 40

# train (adjust paths/GPU count for your environment)
OPENAI_API_KEY=... bash training/exps/run.sh   # see training/README.md for the full flag set used in this reproduction

# evaluate
OPENAI_API_KEY=... python reproduction/scripts/before_after_eval.py --adapter /path/to/checkpoint --n-samples 2
```

`setup.sh` applies all patches described above (except the abandoned vLLM ones in `vllm_attempt/`); no separate patch step is needed.

## Repository map

This repository holds only the added work. `setup.sh` clones upstream TRivia at a pinned commit, applies `patches/trivia-fixes.patch`, and copies `reproduction/` in. Paths that start with `training/` refer to that checkout.

- `reproduction/WRITEUP.md`: this document
- `reproduction/dashboard.html`: reward curve and per-example before/after scatter
- `reproduction/scripts/`: `build_dataset.py`, `before_after_eval.py`, `test_reward_function.py`
- `reproduction/vllm_attempt/`: the unresolved vLLM rollout-server patches, kept as a record
- `reproduction/notebook/`: the full raw Kaggle notebook (code and outputs, no narration; this document is the narrative)
- `patches/trivia-fixes.patch`: the changes to upstream (reward plugin, dataset file, `pt_engine.py`, `grpo_trainer.py`)
- `reward/trivia_reward_plugin.py`: a readable copy of the patched reward plugin, including `TableStructureReward`
