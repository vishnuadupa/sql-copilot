"""
FastAPI + Gradio UI for SQL Copilot.
Inference endpoint + interactive demo.
"""

import os
import sqlite3
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch

# Global model & tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global model, tokenizer
    print("Loading fine-tuned model...")

    # Try to load from HF Hub (set this to your repo after training)
    adapter_path = "vishnuadupa/qwen-sql-lora"

    try:
        print(f"Loading adapter from HF Hub: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")
        model = model.merge_and_unload()
        print("✓ Fine-tuned model loaded!")
    except Exception as e:
        print(f"⚠ Could not load fine-tuned model: {e}")
        print("Falling back to base model...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✓ Base model loaded")

    yield
    print("Unloading model...")
    del model
    torch.cuda.empty_cache()

app = FastAPI(title="SQL Copilot", lifespan=lifespan)

class SQLRequest(BaseModel):
    question: str
    schema: str

class SQLResponse(BaseModel):
    sql: str
    error: str = None

@app.post("/predict", response_model=SQLResponse)
async def predict_sql(req: SQLRequest):
    """Generate SQL from natural language."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Must match train_lora.py's training format exactly (wording + trailing
    # newline after "SQL:") -- this LoRA adapter is lightly trained and
    # very sensitive to prompt shape; a mismatched format was confirmed to
    # collapse accuracy from ~60% to ~2% in eval.py before this same fix.
    prompt = f"""You are a SQL expert. Generate valid SQL.

Question: {req.question}

Schema:
{req.schema}

SQL:
"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
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

        # Take first line only: the model's actual answer is always the
        # first line, and it can drift into commentary (or even a second,
        # unrelated code block) afterward on lightly-trained checkpoints.
        sql = sql.split("\n")[0].strip()

        return SQLResponse(sql=sql)
    except Exception as e:
        return SQLResponse(sql="", error=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

# Sample SQLite database for demo
def create_sample_db():
    """Create a sample DB for testing."""
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

def execute_sql(sql: str):
    """Execute SQL and return results."""
    try:
        conn = sqlite3.connect("sample.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_interface(question, schema):
    """Gradio interface function."""
    if not model:
        return "Model not loaded", "N/A"

    req = SQLRequest(question=question, schema=schema)
    response = execute_sql(req.question)

    # Must match training format exactly -- see note in /predict above.
    prompt = f"""You are a SQL expert. Generate valid SQL.

Question: {req.question}

Schema:
{req.schema}

SQL:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    # Greedy decoding — see note in /predict above.
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True).split("SQL:")[-1].strip()
    # Take first line only — see note in /predict above.
    sql = sql.split("\n")[0].strip()

    try:
        results = execute_sql(sql)
        return sql, json.dumps(results, indent=2)
    except Exception as e:
        return sql, f"Execution error: {str(e)}"

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
                placeholder="Paste your schema here",
                lines=5,
                value="""
Table: customers (id, name, region, revenue)
Table: orders (id, customer_id, amount, date)
                """
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
    create_sample_db()

    # Launch Gradio
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
