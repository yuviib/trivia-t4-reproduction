path = "/kaggle/working/TRivia/training/ms-swift/swift/llm/infer/rollout.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_line = "        engine_kwargs['load_format'] = 'dummy'"
new_line = "        engine_kwargs['load_format'] = 'auto'"

assert old_line in content, "Pattern not found, check file contents"
content = content.replace(old_line, new_line)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched rollout.py load_format successfully")
