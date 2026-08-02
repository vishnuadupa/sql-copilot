"""
Evaluation harness: compare fine-tuned Qwen vs. base Qwen vs. Groq Llama on Spider validation set.
Metric: normalized exact-match accuracy (predicted SQL == gold SQL after normalization).

Note: We use exact-match rather than execution accuracy because execution accuracy
requires the actual per-question Spider SQLite databases, which are large and
Google-Drive-hosted (not available as a lightweight HF mirror). Exact-match is a
standard, widely-reported NL2SQL metric.
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
    """Extract SQL from model response."""
    if "```sql" in text:
        return text.split("```sql")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()


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
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
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


def run_eval(num_samples=50, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", adapter_path="./qwen-sql-lora"):
    print("Loading Spider validation set...")
    dataset = load_dataset("xlangai/spider")
    val_data = dataset["validation"].select(range(min(num_samples, len(dataset["validation"]))))

    print("Loading base Qwen (no fine-tuning)...")
    base_model, base_tokenizer = load_qwen(model_name)

    print("Loading fine-tuned Qwen (LoRA adapter)...")
    ft_model, ft_tokenizer = load_qwen(model_name, adapter_path=adapter_path)

    results = []

    for i, example in enumerate(val_data):
        question = example.get("question", "")
        gold_sql = example.get("query", "")
        db_id = example.get("db_id", "")
        schema = example.get("db_schema", example.get("schema", f"Database: {db_id}"))

        print(f"[{i + 1}/{len(val_data)}] {question[:60]}...")

        pred_base = generate_sql(base_model, base_tokenizer, question, schema)
        pred_ft = generate_sql(ft_model, ft_tokenizer, question, schema)
        pred_groq = eval_groq(question, schema)

        results.append({
            "question": question[:60],
            "base_qwen": int(sql_match(pred_base, gold_sql)),
            "qwen_finetuned": int(sql_match(pred_ft, gold_sql)),
            "groq_llama": int(sql_match(pred_groq, gold_sql)) if pred_groq else 0,
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (normalized exact-match accuracy)")
    print("=" * 60)
    print(f"Base Qwen:       {df['base_qwen'].mean():.2%}")
    print(f"Qwen Fine-tuned: {df['qwen_finetuned'].mean():.2%}")
    print(f"Groq Llama:      {df['groq_llama'].mean():.2%}")

    df.to_csv("eval_results.csv", index=False)
    print("\nResults saved to eval_results.csv")
    return df


if __name__ == "__main__":
    run_eval(num_samples=50)
