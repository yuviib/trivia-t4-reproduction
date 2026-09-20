path = "/kaggle/working/TRivia/training/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_line = "                    state_dict = {k: v for k, v in state_dict.items() if 'visual.' not in k}"
new_line = ("                    state_dict = {k: v for k, v in state_dict.items() if 'visual.' not in k}\n"
            "                    state_dict = {\n"
            "                        (k[len('language_model.'):] if k.startswith('language_model.') else k): v\n"
            "                        for k, v in state_dict.items()\n"
            "                    }")

assert old_line in content, "Pattern not found, check file contents"
content = content.replace(old_line, new_line)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched language_model prefix stripping successfully")
