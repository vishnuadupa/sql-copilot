"""
Evaluation harness: compare fine-tuned Qwen vs. base vs. Groq on Spider validation set.
Metric: execution accuracy (generated SQL runs and returns correct result).
"""

import json
import sqlite3
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from groq import Groq
import os

# Initialize Groq (requires GROQ_API_KEY env var)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

def execute_sql(sql, db_path="sample.db"):
    """Execute SQL and return result set as tuple of tuples."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = tuple(sorted(cursor.fetchall()))
        conn.close()
        return result
    except Exception as e:
        return None

def extract_sql(text):
    """Extract SQL from model response."""
    if "```sql" in text:
        return text.split("```sql")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()

def eval_qwen_base(question, schema):
    """Base Qwen2.5-Coder (no fine-tuning)."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = f"""You are a SQL expert. Given a natural language question and database schema, generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(response.split("SQL:")[-1])

def eval_qwen_finetuned(question, schema, adapter_path="./qwen-sql-lora"):
    """Fine-tuned Qwen with LoRA adapter."""
    from peft import AutoPeftModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")
    model = model.merge_and_unload()

    prompt = f"""You are a SQL expert. Given a natural language question and database schema, generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(response.split("SQL:")[-1])

def eval_groq(question, schema):
    """Groq Llama-3.1-8B via free API."""
    try:
        message = groq_client.messages.create(
            model="llama-3.1-8b-instant",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a SQL expert. Given a natural language question and database schema, generate valid SQL only, no explanation.

Question: {question}

Schema:
{schema}

SQL:"""
                }
            ]
        )
        return extract_sql(message.content[0].text)
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def run_eval(num_samples=50):
    """Run evaluation on Spider validation set."""
    dataset = load_dataset("xlangai/spider")
    val_data = dataset["validation"].select(range(min(num_samples, len(dataset["validation"]))))

    results = []

    for i, example in enumerate(val_data):
        question = example["question"]
        gold_sql = example["query"]
        schema = example["db_schema"]

        print(f"\n[{i+1}/{len(val_data)}] Evaluating: {question[:60]}...")

        # Predictions
        pred_base = eval_qwen_base(question, schema)
        pred_ft = eval_qwen_finetuned(question, schema)
        pred_groq = eval_groq(question, schema)

        # Execution
        gold_result = execute_sql(gold_sql)
        result_base = execute_sql(pred_base) if pred_base else None
        result_ft = execute_sql(pred_ft) if pred_ft else None
        result_groq = execute_sql(pred_groq) if pred_groq else None

        # Accuracy: 1 if execution result matches gold, 0 otherwise
        acc_base = 1 if result_base == gold_result else 0
        acc_ft = 1 if result_ft == gold_result else 0
        acc_groq = 1 if result_groq == gold_result else 0

        results.append({
            "question": question[:60],
            "base_qwen": acc_base,
            "qwen_finetuned": acc_ft,
            "groq_llama": acc_groq,
        })

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(df.describe())
    print("\nModel Accuracy Summary:")
    print(f"Base Qwen: {df['base_qwen'].mean():.2%}")
    print(f"Qwen Fine-tuned: {df['qwen_finetuned'].mean():.2%}")
    print(f"Groq Llama: {df['groq_llama'].mean():.2%}")

    df.to_csv("eval_results.csv", index=False)
    print("\nResults saved to eval_results.csv")

if __name__ == "__main__":
    run_eval(num_samples=50)
