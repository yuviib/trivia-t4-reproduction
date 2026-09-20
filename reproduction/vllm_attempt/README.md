# vLLM rollout-server attempt (not resolved)

The TRivia training recipe generates rollouts on a separate vLLM server and syncs the trainer's updated weights into it after each step. On Kaggle's 2x T4 setup I tried to run that design (`swift rollout` on GPU 1, trainer on GPU 0). It did not work, and the final reproduction uses plain HF `generate()` instead. These scripts are kept as a record of what was tried, not as working code.

They were extracted verbatim from the `%%writefile` cells of [`../notebook/trivia-self-supervised-fine-tuning.ipynb`](../notebook/trivia-self-supervised-fine-tuning.ipynb). Paths are hardcoded to `/kaggle/working/TRivia/...`, and none of these patches are applied by `setup.sh`.

| Script | What it changes | Stated reason (from the code comments) |
|---|---|---|
| `run_rollout_spawn.py` | Launches `rollout_main()` with the `spawn` multiprocessing start method | none recorded |
| `patch_rollout_loadformat.py` | `rollout.py`: `load_format='dummy'` to `'auto'` | none recorded; `patch_grpo_skip_visual.py` assumes the server loads real weights at startup |
| `patch_grpo_skip_visual.py` | `grpo_trainer.py`: skip `visual.*` keys when syncing weights | Frozen vision tower never changes, and vLLM's Qwen2.5-VL loader errors on incremental single-parameter updates for `visual.*` keys |
| `patch_grpo_skip_visual_else.py` | Same filter on a second code path | none recorded |
| `patch_grpo_visual_substring.py` | `startswith('visual.')` to `'visual.' in k` | none recorded |
| `patch_grpo_strip_language_model.py` | Strip the `language_model.` prefix from synced keys | none recorded |

Together the last four are one repeated pattern: adjusting which parameter names get streamed to the server, and how they are named, until the names matched what vLLM's Qwen2.5-VL loader expected.

## Outcome

After these patches the rollout server started and served requests (`Uvicorn running on http://127.0.0.1:8000`), but the weight sync still failed with `KeyError: 'language_model.embed_tokens.weight'` on the server side and the trainer hit `DistNetworkError` and later `CUDA error: device-side assert triggered`. I did not find a root cause for the key mismatch. I switched to HF `generate()` in the trainer, which removed the sync step entirely, and all three reported runs use that path.

I can't say from the notebook whether the `device-side assert` errors were caused by the vLLM path or were an early symptom of the fp16 NaN sampling problem that was later fixed in `pt_engine.py`. Both showed up in the same stretch of debugging.
