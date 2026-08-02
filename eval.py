"""
Evaluation harness: compare fine-tuned Qwen vs. base Qwen vs. Groq Llama.
Metric: normalized exact-match accuracy (predicted SQL == gold SQL after normalization).

Dataset: b-mc2/sql-create-context (includes real CREATE TABLE schema per
example, unlike the Spider parquet mirror which only has a bare db_id).
All three models are given the same real schema in the prompt, so this
is an apples-to-apples comparison.

IMPORTANT: uses the exact same shuffle(seed=42) + slice(range(100)) as
train_lora.py, so this eval set is guaranteed disjoint from the 7000
examples used in training — no leakage.
"""

import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


def extract_sql(text):
    """Extract SQL from model response.

    Take the first non-empty line only. Confirmed via manual inspection
    of raw generations that the model consistently puts its actual answer
    as the first line after "SQL:", then sometimes drifts into commentary
    or even a second, unrelated ```sql``` block afterward — searching for
    a code fence anywhere in the text (the old approach) could grab that
    later, wrong block instead of the real answer.
    """
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    first_line = text.split("\n")[0].strip()
    return first_line


def normalize_sql(sql):
    """Normalize SQL for comparison: lowercase, collapse whitespace, strip trailing semicolon."""
    if not sql:
        return ""
    normalized = " ".join(sql.lower().split())
    return normalized.rstrip(";").strip()


def sql_match(predicted, gold):
    """Normalized exact-match between predicted and gold SQL."""
    return normalize_sql(predicted) == normalize_sql(gold)


def build_prompt(question, schema):
    return f"""You are a SQL expert. Generate valid SQL only, no explanation.

Question: {question}

Schema:
{schema}

SQL:"""


def load_qwen(model_name, adapter_path=None):
    """Load Qwen once (base or with a LoRA adapter merged in)."""
    if adapter_path and os.path.exists(adapter_path):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    return model, tokenizer


def generate_sql(model, tokenizer, question, schema):
    prompt = build_prompt(question, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(response.split("SQL:")[-1])


def eval_groq(question, schema):
    """Llama-3.1-8B via Groq's free API (OpenAI-compatible chat completions)."""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            messages=[{"role": "user", "content": build_prompt(question, schema)}],
        )
        return extract_sql(completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def run_eval(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", adapter_path="./qwen-sql-lora",
             results_path="eval_results.csv"):
    """Runs eval, saving each row to disk as it completes and resuming from
    any partial results_path already on disk. Colab has disconnected mid-run
    more than once this session, and unlike training this loop previously had
    no checkpointing at all -- a disconnect meant re-running all 100 examples
    from zero. Incremental save + resume fixes that.
    """
    print("Loading b-mc2/sql-create-context dataset...")
    raw = load_dataset("b-mc2/sql-create-context")

    # Same seed + same slice as train_lora.py -> guaranteed disjoint from training data
    full = raw["train"].shuffle(seed=42)
    val_data = full.select(range(100))

    results = []
    start_index = 0
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        results = existing_df.to_dict("records")
        start_index = len(results)
        print(f"Found existing {results_path} with {start_index} rows — resuming from example {start_index + 1}.")

    if start_index >= len(val_data):
        print("All examples already evaluated.")
        df = pd.DataFrame(results)
    else:
        print("Loading base Qwen (no fine-tuning)...")
        base_model, base_tokenizer = load_qwen(model_name)

        print("Loading fine-tuned Qwen (LoRA adapter)...")
        ft_model, ft_tokenizer = load_qwen(model_name, adapter_path=adapter_path)

        for i in range(start_index, len(val_data)):
            example = val_data[i]
            question = example["question"]
            gold_sql = example["answer"]
            schema = example["context"]

            print(f"[{i + 1}/{len(val_data)}] {question[:60]}...")

            pred_base = generate_sql(base_model, base_tokenizer, question, schema)
            pred_ft = generate_sql(ft_model, ft_tokenizer, question, schema)
            pred_groq = eval_groq(question, schema)

            row = {
                "question": question[:60],
                "base_qwen": int(sql_match(pred_base, gold_sql)),
                "qwen_finetuned": int(sql_match(pred_ft, gold_sql)),
                "groq_llama": int(sql_match(pred_groq, gold_sql)) if pred_groq else 0,
            }
            results.append(row)

            # Save after every example, not just at the end, so a disconnect
            # loses at most one in-flight example instead of the whole run.
            pd.DataFrame(results).to_csv(results_path, index=False)

        df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (normalized exact-match accuracy)")
    print("Same real schema given to all 3 models; eval set disjoint from training data")
    print("=" * 60)
    print(f"Base Qwen:       {df['base_qwen'].mean():.2%}")
    print(f"Qwen Fine-tuned: {df['qwen_finetuned'].mean():.2%}")
    print(f"Groq Llama:      {df['groq_llama'].mean():.2%}")

    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    return df


if __name__ == "__main__":
    run_eval()
