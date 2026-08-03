"""
FastAPI + Gradio UI for SQL Copilot.
Inference endpoint + interactive demo.
"""

# `spaces` MUST be the first import, before torch/transformers/peft or
# anything that touches CUDA -- confirmed via a hard crash: "RuntimeError:
# CUDA has been initialized before importing the `spaces` package."
# ZeroGPU's own init logic requires it to run before any CUDA-related
# package is even imported (not just used).
import spaces

import os
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch

# Model is loaded once at IMPORT TIME, not inside a FastAPI lifespan hook.
# HF Spaces' Gradio SDK serves the `demo` Blocks object directly and never
# runs `app`'s ASGI lifespan -- confirmed via server logs that this left
# model/tokenizer permanently None, so every request hit the "Model not
# loaded" fallback and crashed the gr.JSON output component with it.
#
# Also: under HF's free ZeroGPU tier, no GPU is visible at import time --
# it's only allocated for the duration of an @spaces.GPU-decorated call.
# So load on CPU here, then move to CUDA inside gradio_interface() below.
print("Loading fine-tuned model...")
ADAPTER_PATH = "vishnuadupa/qwen-sql-lora"

try:
    print(f"Loading adapter from HF Hub: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER_PATH)
    model = model.merge_and_unload()
    print("Fine-tuned model loaded.")
except Exception as e:
    print(f"Could not load fine-tuned model ({e}); falling back to base model.")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.float16
    )
    print("Base model loaded.")

model = model.to("cpu")

app = FastAPI(title="SQL Copilot")


class SQLRequest(BaseModel):
    question: str
    schema: str


class SQLResponse(BaseModel):
    sql: str
    error: str = None


def build_prompt(question: str, schema: str) -> str:
    # Must match train_lora.py's training format exactly (wording + trailing
    # newline after "SQL:") -- this LoRA adapter is lightly trained and
    # very sensitive to prompt shape; a mismatched format was confirmed to
    # collapse accuracy from ~60% to ~2% in eval.py before this same fix.
    return f"""You are a SQL expert. Generate valid SQL.

Question: {question}

Schema:
{schema}

SQL:
"""


def generate_sql_text(question: str, schema: str, device: str = "cpu") -> str:
    prompt = build_prompt(question, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Greedy decoding, not sampling: a SQL generator should return its
    # single most-confident answer, not a random draw that can vary
    # between identical requests.
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = response.split("SQL:")[-1].strip()

    if "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()

    # Take first line only: the model's actual answer is always the first
    # line, and it can drift into commentary (or even a second, unrelated
    # code block) afterward on lightly-trained checkpoints.
    return sql.split("\n")[0].strip()


@app.post("/predict", response_model=SQLResponse)
async def predict_sql(req: SQLRequest):
    """Generate SQL from natural language."""
    try:
        sql = generate_sql_text(req.question, req.schema)
        return SQLResponse(sql=sql)
    except Exception as e:
        return SQLResponse(sql="", error=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


def create_sample_db():
    """Create a sample DB for the demo."""
    conn = sqlite3.connect("sample.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            revenue REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount REAL,
            date TEXT
        )
    """)

    cursor.execute("DELETE FROM customers")
    cursor.execute("DELETE FROM orders")

    customers = [
        (1, "Acme Corp", "Northeast", 50000),
        (2, "TechStart", "West", 75000),
        (3, "BigCorp", "Northeast", 120000),
        (4, "SmallBiz", "South", 30000),
    ]
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

    orders = [
        (1, 1, 5000, "2024-01-15"),
        (2, 1, 3000, "2024-02-10"),
        (3, 3, 15000, "2024-01-20"),
        (4, 2, 8000, "2024-03-05"),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()


# Run unconditionally at import time -- HF Spaces' Gradio SDK imports this
# module rather than executing it as __main__, so anything gated behind
# `if __name__ == "__main__":` never runs there.
create_sample_db()


def execute_sql(sql: str):
    """Execute SQL, returning (results_or_None, error_message_or_None)."""
    try:
        conn = sqlite3.connect("sample.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results, None
    except Exception as e:
        return None, str(e)


@spaces.GPU
def gradio_interface(question, schema):
    """Gradio interface function.

    @spaces.GPU is required by HF Spaces' free ZeroGPU tier -- without it,
    the Space fails at startup with "No @spaces.GPU function detected"
    since ZeroGPU only allocates GPU time to functions explicitly marked
    this way. GPU is only actually visible for the duration of this call,
    so the model is moved to "cuda" here rather than at import time.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        sql = generate_sql_text(question, schema, device=device)
    except Exception as e:
        # result_output is a gr.JSON component: it always requires a
        # JSON-serializable value, never a bare string, or it crashes with
        # "Invalid JSON string" -- confirmed via server logs to be exactly
        # what was happening on every request before this fix.
        return "", json.dumps({"error": f"Generation failed: {e}"})

    results, error = execute_sql(sql)
    if error:
        return sql, json.dumps({"error": error})
    return sql, json.dumps(results, indent=2)


# Gradio UI
with gr.Blocks(title="SQL Copilot") as demo:
    gr.Markdown("# SQL Copilot — NL → SQL")
    gr.Markdown("Fine-tuned Qwen2.5-Coder on Spider dataset. Ask a question, get SQL.")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g., 'Top 5 customers by revenue in Northeast'",
                lines=2
            )
            schema = gr.Textbox(
                label="Database Schema",
                placeholder="Paste your schema here (CREATE TABLE statements)",
                lines=5,
                # CREATE TABLE format, not a "Table: x (...)" summary --
                # matches what the model was actually trained on
                # (b-mc2/sql-create-context), so the default example
                # actually demonstrates the model's real behavior.
                value="""CREATE TABLE customers (id INTEGER, name TEXT, region TEXT, revenue REAL)
CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL, date TEXT)"""
            )
            submit_btn = gr.Button("Generate SQL", variant="primary")

        with gr.Column():
            sql_output = gr.Code(label="Generated SQL", language="sql")
            result_output = gr.JSON(label="Query Results")

    submit_btn.click(
        fn=gradio_interface,
        inputs=[question, schema],
        outputs=[sql_output, result_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
