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
import os
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# QUICK_TEST=1 runs a fast ~200-example, 1-epoch sanity check in its own
# output dir/Hub-less mode — proves the EOS-stop fix actually works before
# committing to the full ~36 min / 7000-example run. It never touches the
# real output_dir (Drive-persisted) or pushes to the real Hub model.
QUICK_TEST = os.environ.get("QUICK_TEST") == "1"

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
if QUICK_TEST:
    print("QUICK_TEST=1: using 200 training examples, 1 epoch, separate output dir")
    train_dataset_raw = full.select(range(100, 300))
else:
    train_dataset_raw = full.select(range(100, 7100))

def format_prompt(example):
    """Format training example with real schema.

    Appends the tokenizer's EOS token after the SQL answer. Without this,
    the model never learns where an answer ends and — confirmed by
    inspecting raw generations — keeps appending plausible-looking but
    wrong extra clauses (ORDER BY/LIMIT/GROUP BY) after a correct answer,
    since nothing in training ever taught it to stop.
    """
    question = example["question"]
    schema = example["context"]
    sql = example["answer"]

    prompt = f"""You are a SQL expert. Generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:
{sql}"""
    return {"text": prompt + tokenizer.eos_token}

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

# QUICK_TEST uses its own small-scale schedule: 200 examples at effective
# batch size 8 is ~25 total steps, so the real run's eval_steps/save_steps
# (100) would never fire at all (0 evaluations logged), and
# load_best_model_at_end would then have no eval history to pick from.
if QUICK_TEST:
    quick_output_dir = "./quick-test-lora"
    training_args = SFTConfig(
        output_dir=quick_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["lr"]),
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        eval_steps=10,
        save_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        push_to_hub=False,
        fp16=True,
        optim="paged_adamw_32bit",
        report_to=[],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )
else:
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

import shutil
import hashlib

latest_checkpoint = None
final_output_dir = quick_output_dir if QUICK_TEST else cfg["training"]["output_dir"]

if QUICK_TEST:
    # Quick test always starts fresh in its own throwaway dir — no
    # resume logic, no signature tracking, and it never touches the
    # real (Drive-persisted) checkpoint_dir used by the full run.
    print(f"QUICK_TEST=1: fresh run in {final_output_dir}, not touching the real checkpoint dir")
else:
    # Auto-resume from latest checkpoint, but ONLY if it was produced by this
    # exact dataset/config (a checkpoint from a previous run with different
    # training data must never be silently resumed into — the run tags its
    # output dir with a signature file so a stale checkpoint from an older
    # schema/dataset version is detected and ignored instead of corrupting
    # this run).
    checkpoint_dir = cfg["training"]["output_dir"]
    # v2: added EOS token after each training example so the model learns to
    # stop instead of appending extra clauses after a correct answer. Bumping
    # this tag forces a fresh run instead of resuming into the old v1
    # checkpoint, which was trained without that fix.
    run_signature = hashlib.sha256(
        f"b-mc2/sql-create-context|seed=42|train=100:7100|v2-eos-fix".encode()
    ).hexdigest()[:16]
    signature_path = os.path.join(checkpoint_dir, "RUN_SIGNATURE.txt")

    def checkpoint_step(dirname):
        """Extract the numeric step count from a 'checkpoint-N' folder name.

        NOTE: plain sorted(list_of_names) sorts as text, so 'checkpoint-900'
        comes after 'checkpoint-2600' alphabetically even though 900 < 2600.
        Sorting by this numeric key instead is what actually picks the latest
        checkpoint once step counts cross a digit-length boundary (e.g. 999 -> 1000).
        """
        return int(dirname.split("-")[-1])

    if os.path.exists(checkpoint_dir):
        existing_signature = None
        if os.path.exists(signature_path):
            with open(signature_path) as f:
                existing_signature = f.read().strip()

        if existing_signature != run_signature:
            print("Found checkpoint dir from a different run (signature mismatch) — "
                  "deleting it and starting fully fresh (old checkpoints are from "
                  "training that used different data/format and must never be mixed "
                  "in with this run's step numbering).")
            # checkpoint_dir may be a symlink (e.g. pointed at a Google Drive
            # folder so checkpoints survive Colab disconnects). shutil.rmtree()
            # refuses to operate on a symlink directly — it raises OSError
            # rather than silently deleting through it — so resolve to the
            # real target first and clear that, leaving the symlink itself intact.
            real_dir = os.path.realpath(checkpoint_dir) if os.path.islink(checkpoint_dir) else checkpoint_dir
            shutil.rmtree(real_dir)
            # Recreate the real target explicitly rather than relying on a later
            # os.makedirs(checkpoint_dir) to correctly follow a now-dangling
            # symlink and materialize the directory at its target — that
            # behavior isn't guaranteed portable, so do it directly here.
            os.makedirs(real_dir, exist_ok=True)
        else:
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = os.path.join(
                    checkpoint_dir, max(checkpoints, key=checkpoint_step)
                )
                print(f"Resuming from checkpoint: {latest_checkpoint}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(signature_path, "w") as f:
        f.write(run_signature)

# Trainer
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

trainer.train(resume_from_checkpoint=latest_checkpoint)

print("Saving model...")
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

if QUICK_TEST:
    print(f"Quick test complete. Model saved locally to {final_output_dir} (not pushed to Hub).")
else:
    print("Training complete. Model saved and pushed to HF Hub.")
