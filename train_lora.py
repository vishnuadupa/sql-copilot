"""
LoRA fine-tuning Qwen2.5-Coder on Spider dataset.
Run on Colab with free T4 GPU. ~2 hours total.
"""

import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from unsloth import FastLanguageModel

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

# LoRA config
peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    target_modules=cfg["lora"]["target_modules"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Load dataset
print("Loading Spider dataset...")
dataset = load_dataset("yale-lily/spider")

def format_prompt(example):
    """Format NL question + schema → SQL instruction."""
    question = example["question"]
    sql = example["query"]
    schema = example["db_schema"]
    prompt = f"""You are a SQL expert. Given a natural language question and database schema, generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:
{sql}"""
    return {"text": prompt}

train_dataset = dataset["train"].map(format_prompt)
eval_dataset = dataset["validation"].map(format_prompt).select(range(100))  # 100 for quick eval

# Training args
training_args = TrainingArguments(
    output_dir=cfg["training"]["output_dir"],
    num_train_epochs=cfg["training"]["epochs"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    per_device_eval_batch_size=cfg["training"]["batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["lr"],
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=True,
    hub_model_id=cfg["training"]["hub_model_id"],
    hub_strategy="every_save",
    fp16=True,
    optim="paged_adamw_32bit",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    args=training_args,
    packing=False,
    max_seq_length=max_seq_length,
)

print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained(cfg["training"]["output_dir"])
tokenizer.save_pretrained(cfg["training"]["output_dir"])

print("Training complete. Model saved and pushed to HF Hub.")
