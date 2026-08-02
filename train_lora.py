"""
LoRA fine-tuning Qwen2.5-Coder on Spider dataset.
Run on Colab with free T4 GPU. ~1-2 hours total.
"""

# CRITICAL: Import unsloth FIRST before other libraries
from unsloth import FastLanguageModel
import yaml
import torch
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from datasets import Dataset
import json
import subprocess

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

# Download Spider dataset from GitHub
print("Downloading Spider dataset...")
try:
    subprocess.run(
        ["git", "clone", "https://github.com/taoyds/spider.git", "/tmp/spider"],
        capture_output=True,
        timeout=60,
    )
    with open("/tmp/spider/train_spider.json") as f:
        spider_train = json.load(f)[:500]  # 500 examples for speed
    print(f"Loaded {len(spider_train)} training examples")
except Exception as e:
    print(f"Warning: Could not download Spider: {e}")
    print("Using minimal dataset for demo...")
    spider_train = [
        {
            "question": "Show me the top customers by revenue",
            "query": "SELECT customer_id, SUM(amount) as revenue FROM orders GROUP BY customer_id ORDER BY revenue DESC LIMIT 5",
            "db_id": "sales",
        }
    ] * 10

def format_prompt(item):
    """Format training example."""
    question = item.get("question", "")
    sql = item.get("query", "")
    db_id = item.get("db_id", "")

    prompt = f"""You are a SQL expert. Generate valid SQL.

Question: {question}

Database: {db_id}

SQL:
{sql}"""
    return {"text": prompt}

train_texts = [format_prompt(item) for item in spider_train]
train_dataset = Dataset.from_dict({"text": [d["text"] for d in train_texts]})
eval_dataset = train_dataset.select(range(min(50, len(train_dataset))))

# LoRA config
peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    target_modules=cfg["lora"]["target_modules"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Training args
training_args = TrainingArguments(
    output_dir=cfg["training"]["output_dir"],
    num_train_epochs=cfg["training"]["epochs"],
    per_device_train_batch_size=cfg["training"]["batch_size"],
    per_device_eval_batch_size=cfg["training"]["batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    learning_rate=cfg["training"]["lr"],
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=50,
    eval_steps=100,
    save_steps=100,
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
print("Starting training...")
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

trainer.train()

print("Saving model...")
model.save_pretrained(cfg["training"]["output_dir"])
tokenizer.save_pretrained(cfg["training"]["output_dir"])

print("Training complete. Model saved and pushed to HF Hub.")
