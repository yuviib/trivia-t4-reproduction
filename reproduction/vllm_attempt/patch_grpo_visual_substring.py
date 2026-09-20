path = "/kaggle/working/TRivia/training/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

count = content.count("if not k.startswith('visual.')")
content = content.replace(
    "state_dict = {k: v for k, v in state_dict.items() if not k.startswith('visual.')}",
    "state_dict = {k: v for k, v in state_dict.items() if 'visual.' not in k}"
)
content = content.replace(
    "if name.startswith('visual.'):",
    "if 'visual.' in name:"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Replaced {count} occurrence(s) of the startswith filter with substring filter")
