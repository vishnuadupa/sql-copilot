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
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

ADAPTER_PATH = "vishnuadupa/qwen-sql-lora"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Model is loaded LAZILY, on first use, from inside the @spaces.GPU-decorated
# gradio_interface() -- not at import time, and not in a FastAPI lifespan
# hook (HF Spaces' Gradio SDK serves the `demo` Blocks object directly and
# never runs `app`'s ASGI lifespan, confirmed via logs: model/tokenizer
# stayed permanently None, so every request hit "Model not loaded").
#
# Loading at plain module level doesn't work either: `spaces` installs a
# global torch patch the moment it's imported that intercepts ALL tensor
# ops -- including a plain CPU safetensors.load_file() call -- and routes
# them through a GPU-context check. Confirmed via full traceback: loading
# the adapter at import time crashed inside that patch with "No CUDA GPUs
# are available", regardless of dtype/quantization choices, simply
# because no @spaces.GPU call was active yet. So loading must happen
# inside the decorated function itself, the first time it's actually
# invoked (which IS a valid GPU-allocated context under ZeroGPU).
_model = None
_tokenizer = None


def _ensure_model_loaded():
    global _model, _tokenizer
    if _model is not None:
        return
    print("Loading fine-tuned model...")
    try:
        print(f"Loading adapter from HF Hub: {ADAPTER_PATH}")
        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32)
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model = _model.merge_and_unload()
        print("Fine-tuned model loaded.")
    except Exception as e:
        import traceback
        print(f"Could not load fine-tuned model ({e}); falling back to base model.")
        traceback.print_exc()
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32)
        print("Base model loaded.")


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
    _ensure_model_loaded()
    prompt = build_prompt(question, schema)
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)
    # Greedy decoding, not sampling: a SQL generator should return its
    # single most-confident answer, not a random draw that can vary
    # between identical requests.
    outputs = _model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.eos_token_id,
    )
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    """Execute SQL, returning (results_or_None, error_message_or_None).

    Results are returned as a list of {column: value} dicts (using
    cursor.description for column names) rather than bare tuples/lists --
    much more readable in the JSON output (e.g. {"name": "TechStart"}
    instead of an unlabeled ["TechStart"]).
    """
    try:
        conn = sqlite3.connect("sample.db")
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        results = [dict(zip(columns, row)) for row in rows]
        return results, None
    except Exception as e:
        return None, str(e)


# Verified working (question, schema) pairs for the "Try an Example" button.
# Each was manually tested against the live deployed model before being
# added here, rather than guessed -- every one below produced a correct,
# sensible query + result when tested.
DEFAULT_SCHEMA = """CREATE TABLE customers (id INTEGER, name TEXT, region TEXT, revenue REAL)
CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL, date TEXT)"""

EXAMPLES = [
    {
        "question": "Top 5 customers by revenue in Northeast",
        "schema": DEFAULT_SCHEMA,
    },
    {
        "question": "List all customers in the West region",
        "schema": DEFAULT_SCHEMA,
    },
]


def pick_random_example():
    example = random.choice(EXAMPLES)
    return example["question"], example["schema"]


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

    try:
        _ensure_model_loaded()  # first call here loads inside a valid GPU context
        _model.to(device)
        sql = generate_sql_text(question, schema, device=device)
    except Exception as e:
        import traceback
        print(f"gradio_interface generation error: {e}")
        traceback.print_exc()
        # result_output is a gr.JSON component: it always requires a
        # JSON-serializable value, never a bare string, or it crashes with
        # "Invalid JSON string" -- confirmed via server logs to be exactly
        # what was happening on every request before this fix.
        return "", json.dumps({"error": f"Generation failed: {e}"})

    results, error = execute_sql(sql)
    if error:
        return sql, json.dumps({"error": error})
    if not results:
        return sql, json.dumps({"info": "Query executed successfully — no matching rows."})
    return sql, json.dumps(results, indent=2)


# Gradio UI
with gr.Blocks(title="SQL Copilot", theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🗄️ SQL Copilot — Natural Language → SQL")
    gr.Markdown(
        "Fine-tuned Qwen2.5-Coder-1.5B (LoRA) on the b-mc2/sql-create-context dataset. "
        "Type a question and schema, or click **Try an Example** for one that's verified to work well.\n\n"
        "*Honest note: this is a small, lightly-fine-tuned model evaluated on unseen schemas — it "
        "sometimes gets things exactly right and sometimes hallucinates a filter condition that wasn't "
        "asked for. Both are shown here on purpose.*"
    )

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g., 'Top 5 customers by revenue in Northeast'",
                lines=2
            )
            schema = gr.Code(
                label="Database Schema (CREATE TABLE statements)",
                language="sql",
                lines=5,
                value=DEFAULT_SCHEMA,
            )
            with gr.Row():
                example_btn = gr.Button("🎲 Try an Example")
                submit_btn = gr.Button("Generate SQL", variant="primary")
            gr.Markdown(
                "*First request after the app has been idle takes ~15-20s "
                "(free-tier GPU has to reload the model); after that it's fast.*"
            )

        with gr.Column():
            sql_output = gr.Code(label="Generated SQL", language="sql")
            result_output = gr.JSON(label="Query Results")

    example_btn.click(
        fn=pick_random_example,
        inputs=[],
        outputs=[question, schema]
    )

    submit_btn.click(
        fn=gradio_interface,
        inputs=[question, schema],
        outputs=[sql_output, result_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
