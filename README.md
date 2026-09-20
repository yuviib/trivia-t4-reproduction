# TRivia reproduction on a single free T4

An attempt to reproduce the training loop from **TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition** ([arXiv:2512.01248](https://arxiv.org/abs/2512.01248), CVPR 2026) on one free Kaggle T4, using LoRA on Qwen2.5-VL-3B instead of the paper's 8-GPU full fine-tuning.

## Result in one paragraph

The GRPO loop runs end to end once four bugs in the released code are fixed, and the reward functions behave as intended on real data. It did not produce a demonstrated improvement. On the 40 training examples the LoRA checkpoint scored lower than base on QA_F1 (0.196 to 0.107; 4 examples up, 11 down, 25 unchanged), but with 2 samples per example at temperature 1.0 through an LLM judge, that difference is not statistically distinguishable from noise (sign test p = 0.12). The QA_F1 reward curve also shows no clear upward trend within the run. So this shows a working pipeline, not a trained model that got better.

## Contents

| Path | What it is |
|---|---|
| [reproduction/WRITEUP.md](./reproduction/WRITEUP.md) | The full account: scope, bugs, extensions, results, limitations, next steps |
| [reproduction/dashboard.html](./reproduction/dashboard.html) | Reward curve and per-example before/after scatter (open in a browser) |
| [reproduction/scripts/](./reproduction/scripts) | Dataset builder, reward-function test, and the before/after evaluation |
| [reproduction/vllm_attempt/](./reproduction/vllm_attempt/README.md) | The vLLM rollout-server patches that did not work, kept as a record |
| [reproduction/notebook/](./reproduction/notebook) | The raw Kaggle notebook, code and outputs |
| [patches/trivia-fixes.patch](./patches/trivia-fixes.patch) | All changes to upstream, applied by `setup.sh` |
| [reward/trivia_reward_plugin.py](./reward/trivia_reward_plugin.py) | Readable copy of the patched reward plugin, including the new `TableStructureReward` |

This repository does not contain the upstream code. `setup.sh` fetches it.

## What was changed in upstream

- Reward plugin: fixed the OpenAI client wiring and an import path, and added `TableStructureReward`, a label-free structural reward that is not in the paper.
- Dataset file: the prompt asked for OTSL, a format the base model has never seen, which made every reward silently zero. It now asks for HTML, with image paths fixed.
- `pt_engine.py`: added a NaN-safe logits processor to fix an fp16 sampling crash on GPUs without bf16 (such as the T4).
- `grpo_trainer.py`: completion logging was a silent no-op because the vendored trainer deleted the log method. It now writes through the trainer's existing writer.

## Quickstart

```bash
./setup.sh            # creates ./TRivia: upstream at a pinned commit, plus the patch, plus reproduction/
cd TRivia

pip install -e training/ms-swift
pip install "qwen_vl_utils[decord]<0.0.12" apted math_verify==0.5.2 jieba openai python-Levenshtein

# optional: expand the training set beyond the bundled 10 examples
python reproduction/scripts/build_dataset.py --n 40

# train (see training/README.md and reproduction/WRITEUP.md for the flags used)
OPENAI_API_KEY=... bash training/exps/run.sh

# evaluate a checkpoint against the base model
OPENAI_API_KEY=... python reproduction/scripts/before_after_eval.py \
    --adapter /path/to/checkpoint --n-samples 2

# to check whether a delta is real rather than sampling noise, add
# --seed 0 --n-samples 8, or --temperature 0 (greedy); --out results.json saves per-row scores
```

The trained LoRA checkpoint is not included. The evaluation script has not been re-run since its last edit (seed, temperature and paired-statistics options); the recorded results come from the notebook.

## Citation

Please cite the original paper:

```
@misc{zhang2025triviaselfsupervisedfinetuningvisionlanguage,
      title={TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition},
      author={Junyuan Zhang and Bin Wang and Qintong Zhang and Fan Wu and Zichen Wen and Jialin Lu and Junjie Shan and Ziqi Zhao and Shuya Yang and Ziling Wang and Ziyang Miao and Huaping Zhong and Yuhang Zang and Xiaoyi Dong and Ka-Ho Chow and Conghui He},
      year={2025},
      eprint={2512.01248},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.01248},
}
```

## License

[Apache License 2.0](./LICENSE), the same as upstream. See [NOTICE](./NOTICE) for attribution.

The base model, Qwen2.5-VL-3B-Instruct, has its own license (not Apache 2.0); check its terms before commercial use. Training data built by `build_dataset.py` comes from `terryoo/TableVQA-Bench` (fintabnetqa split), derived from FinTabNet (CDLA-Permissive).
