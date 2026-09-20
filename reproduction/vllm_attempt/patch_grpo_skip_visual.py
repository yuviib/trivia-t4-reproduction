path = "/kaggle/working/TRivia/training/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = """                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                    if parameter_group_no_lora:"""

new_block = """                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                    # Frozen vision tower/merger weights never change during training and are already
                    # correctly loaded on the rollout server at startup (load_format=auto). Skip
                    # streaming them: vLLM's Qwen2.5-VL loader errors on incremental single-parameter
                    # updates for visual.* keys.
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('visual.')}
                    if parameter_group_no_lora:"""

assert old_block in content, "Pattern not found, check file contents"
content = content.replace(old_block, new_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched grpo_trainer.py successfully")
