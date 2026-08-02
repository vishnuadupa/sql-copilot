"""
LoRA fine-tuning Qwen2.5-Coder on NL-to-SQL data with real schema context.
Run on Colab with free T4 GPU. ~1-2 hours total.

Dataset: b-mc2/sql-create-context — each example includes the actual
CREATE TABLE schema text (not just a database name), so the model
learns to read a schema, not memorize specific database names.
"""

# CRITICAL: Import unsloth FIRST before other libraries
from unsloth import FastLanguageModel
import yaml
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Model setup
model_name = cfg["model"]["name"]
max_seq_length = cfg["model"]["max_seq_length"]
dtype = torch.float16

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# Load dataset with real schema context
print("Loading b-mc2/sql-create-context dataset...")
raw = load_dataset("b-mc2/sql-create-context")

# Shuffle once with a fixed seed, then split deterministically:
# first 100 examples -> eval.py's held-out set (NEVER used in training)
# next 7000 examples -> training set
# Same seed + same slicing must be used in eval.py to guarantee zero overlap.
full = raw["train"].shuffle(seed=42)
eval_dataset_raw = full.select(range(100))
train_dataset_raw = full.select(range(100, 7100))

def format_prompt(example):
    """Format training example with real schema."""
    question = example["question"]
    schema = example["context"]
    sql = example["answer"]

    prompt = f"""You are a SQL expert. Generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:
{sql}"""
    return {"text": prompt}

train_dataset = train_dataset_raw.map(format_prompt)
eval_dataset = eval_dataset_raw.map(format_prompt)

print(f"Training examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)} (held out, disjoint from training)")

# Attach LoRA adapters to the quantized model (must happen before the trainer
# is built — SFTTrainer's own peft_config= path does not work on a 4-bit model)
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    target_modules=cfg["lora"]["target_modules"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Training args — using SFTConfig (modern TRL API bundles seq-length/packing here,
# not in the trainer constructor, and field names shift between TRL versions)
training_args = SFTConfig(
    output_dir=cfg["training"]["output_dir"],
    num_train_epochs=cfg["training"]["epochs"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    per_device_eval_batch_size=cfg["training"]["batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=float(cfg["training"]["lr"]),
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=50,
    eval_steps=cfg["training"]["eval_steps"],
    save_steps=cfg["training"]["save_steps"],
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=cfg["training"]["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_model_id=cfg["training"]["hub_model_id"],
    hub_strategy="every_save",
    fp16=True,
    optim="paged_adamw_32bit",
    report_to=[],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
)

# Trainer
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

# Auto-resume from latest checkpoint if interrupted
import os
checkpoint_dir = cfg["training"]["output_dir"]
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
        print(f"Resuming from checkpoint: {latest_checkpoint}")

trainer.train(resume_from_checkpoint=latest_checkpoint)

print("Saving model...")
model.save_pretrained(cfg["training"]["output_dir"])
tokenizer.save_pretrained(cfg["training"]["output_dir"])

print("Training complete. Model saved and pushed to HF Hub.")
