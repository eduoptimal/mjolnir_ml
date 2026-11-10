"""mjolnir_rlvf.train — DPO over typed preference pairs (the M2 stage).

Distilled from the research `rlvf/dpo_train.py` (the run that produced the
1.5B corrective adapter). Heavy dependencies are LAZY: importing this module
never imports torch; calling `train()` without the extra fails with the
install line, never a traceback. Reward: none here — the pairs were already
scored by the authority during rollout; this stage only optimizes them.
"""
from __future__ import annotations

import os

_MISSING = ("RLVF train needs the training extra: "
            'pip install "mjolnir[rlvf]"  (torch, transformers, datasets, '
            "peft, trl)")


def train(data_dir: str, model_path: str, out_dir: str, *,
          epochs: float = 2.0, seed: int = 42, lr: float = 5e-6,
          beta: float = 0.1, max_length: int = 512,
          lora_r: int = 8, lora_alpha: int = 16,
          lora_dropout: float = 0.05,
          batch_size: int = 1, grad_accum: int = 4) -> str:
    """DPO-train `model_path` on data_dir/{train,valid}.jsonl -> out_dir.

    The pair layout is the contract pairs.write_pairs already produces
    ({prompt, chosen, rejected, ...}); the hyperparameter defaults are the
    measured ones from the 1.5B corrective run (small sets, ~9 steps: seed
    variance is expected and recorded, not hidden).
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise RuntimeError(_MISSING) from e

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32).to(device)
    ds = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.jsonl"),
        "valid": os.path.join(data_dir, "valid.jsonl")})

    def fmt(ex):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        return {"prompt": prompt, "chosen": ex["chosen"],
                "rejected": ex["rejected"]}

    ds = ds.map(fmt, remove_columns=ds["train"].column_names)
    print(f"pairs: train={len(ds['train'])} valid={len(ds['valid'])} "
          f"device={device}")
    peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj"],
                          task_type="CAUSAL_LLM")
    args = DPOConfig(output_dir=out_dir,
                     per_device_train_batch_size=batch_size,
                     gradient_accumulation_steps=grad_accum,
                     num_train_epochs=epochs, learning_rate=lr, beta=beta,
                     max_length=max_length, logging_steps=5,
                     save_strategy="no", report_to=[], fp16=False,
                     bf16=False, seed=seed, data_seed=seed)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds["train"],
                         eval_dataset=ds["valid"], processing_class=tok,
                         peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(out_dir)
    print("DPO_DONE", out_dir)
    return out_dir
