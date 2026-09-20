path = "/kaggle/working/TRivia/training/ms-swift/swift/trainers/rlhf_trainer/grpo_trainer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = """        else:
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == 'colocate':
                        llm_model = self.engine.inner_model
                        llm_model.load_weights([(name, param.data)])"""

new_block = """        else:
            for name, param in self.model.named_parameters():
                if name.startswith('visual.'):
                    continue
                with gather_if_zero3([param]):
                    if self.vllm_mode == 'server' and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)
                    elif self.vllm_mode == 'colocate':
                        llm_model = self.engine.inner_model
                        llm_model.load_weights([(name, param.data)])"""

assert old_block in content, "Pattern not found, check file contents"
content = content.replace(old_block, new_block)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched else-branch successfully")
