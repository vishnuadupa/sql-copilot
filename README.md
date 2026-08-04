# SQL Copilot: Fine-Tuned NL → SQL with Qwen2.5-Coder

A small (1.5B param) LoRA-fine-tuned model that translates natural language questions into SQL, deployed live on HuggingFace Spaces.

🔗 [Live Demo](https://huggingface.co/spaces/vishnuadupa/sql-copilot) · [Fine-tuned Model](https://huggingface.co/vishnuadupa/qwen-sql-lora)

---

## Overview

This project fine-tunes Qwen2.5-Coder-1.5B via LoRA to generate SQL from a natural-language question plus a database schema, then deploys it as a live, interactive demo. It includes a leak-free evaluation harness comparing the fine-tuned model against the untrained base model and a Groq-hosted Llama-3.1-8B baseline.

Measured accuracy is ~2% normalized exact-match on 100 held-out, unseen-schema questions — low in absolute terms, for reasons explained below, but a legitimate end-to-end ML result: real training, real held-out evaluation, real deployment.

---

## Results

Measured on 100 examples held out from training (disjoint via a shared `shuffle(seed=42)` split), using greedy decoding and a prompt format identical to training:

| Model | Normalized exact-match accuracy |
|---|---|
| Qwen2.5-Coder-1.5B (fine-tuned) | ~2% |
| Qwen2.5-Coder-1.5B (base, no fine-tuning) | ~1% |
| Llama-3.1-8B (Groq, zero-shot) | ~0% |

**Why this number is low:**

1. **Strict metric.** Exact-string-match fails a query that's functionally identical but differently formatted — reordered `WHERE` clauses, different quoting, a missing/extra aggregate all score as wrong even when the SQL logic is correct.
2. **Tiny adapter.** Only 0.14% of the model's parameters were trained (LoRA rank 16 on 2 projection matrices).
3. **Genuine held-out generalization.** The evaluation set uses table/column names the model never saw during training — a real generalization test, not memorization.
4. **Consistent with live behavior.** The deployed model sometimes generates exactly correct joins/filters/aggregates, and sometimes hallucinates a filter condition that wasn't asked for. Both are shown deliberately in the live demo's UI.

---

## Architecture

```
b-mc2/sql-create-context (7k examples, real CREATE TABLE schema per example)
    │
    ▼
LoRA fine-tune on Qwen2.5-Coder-1.5B-Instruct (Colab free T4, ~35 min)
    │  checkpoints persisted to Google Drive (survives Colab disconnects)
    ▼
Evaluation harness (eval.py)
    │  disjoint held-out split, greedy decoding, signature-checked resume
    ▼
Deployment: FastAPI + Gradio on HuggingFace Spaces (free ZeroGPU tier)
    │  model loaded lazily inside the GPU-decorated call (ZeroGPU requirement)
    ▼
Live demo: https://huggingface.co/spaces/vishnuadupa/sql-copilot
```

---

## Technical Details

### Data
- **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — includes real `CREATE TABLE` schema text per example, not just a bare database name
- **Split:** `full.shuffle(seed=42)`; first 100 examples → eval set, next 7,000 → training set. Same seed used in both `train_lora.py` and `eval.py`, guaranteeing zero overlap.

### Fine-tuning
- **Base model:** Qwen2.5-Coder-1.5B-Instruct (Apache 2.0)
- **Method:** LoRA, rank 16, alpha 32, targeting `q_proj`/`v_proj`, dropout 0.05 (2,179,072 trainable params — 0.14% of the model)
- **Training text includes an explicit EOS token** after each SQL answer, so the model learns where an answer ends instead of appending extra clauses afterward
- **Hardware:** Colab free T4 (~35 min for 3 epochs / 2,625 steps), checkpoints on Google Drive so a Colab disconnect never loses progress
- **Config:** see `config.yaml` (Colab) / `config.local.yaml` (local GPU training, tuned for 4GB VRAM)

### Evaluation
- **Metric:** normalized exact-match (lowercased, whitespace-collapsed, trailing semicolon stripped)
- **Decoding:** greedy (`do_sample=False`) for reproducible scoring
- **Resume-safe:** results save incrementally and are tagged with a model+adapter signature, so a stale results file is never mistaken for a completed run

### Deployment
- **Stack:** FastAPI (`/predict` API) + Gradio (interactive UI), served together from `app.py`
- **Hosting:** HuggingFace Spaces, free ZeroGPU tier
- **Known limitation:** ZeroGPU doesn't persist Python process state between GPU-decorated calls, so the model reloads from HF Hub on every request (~10-20s overhead). This is architectural to the free tier.
- **Free tier also has a daily GPU-time quota** — the demo may occasionally show a GPU-acquisition error until the quota resets.

---

## Quick Start

### Local setup
```bash
git clone https://github.com/vishnuadupa/sql-copilot.git
cd sql-copilot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train (Colab, free T4 GPU)
```python
!git clone https://github.com/vishnuadupa/sql-copilot.git
%cd sql-copilot
!pip install -q -r requirements-train.txt

from huggingface_hub import login
login(token="your_hf_token")  # https://huggingface.co/settings/tokens

!python train_lora.py
```
`config.yaml`'s `hub_model_id` points at a pushed model — change it to your own HF username to train your own version.

### Evaluate
```bash
export GROQ_API_KEY="your_groq_api_key"   # free tier, no card required
python eval.py
```
Outputs `eval_results.csv` with per-question, per-model correctness.

### Run the app locally
```bash
python app.py
```
Opens a Gradio UI at `http://localhost:7860`.

### Deploy to HuggingFace Spaces
1. Create a Space at https://huggingface.co/new-space with **SDK: Gradio**
2. Push `app.py` + `requirements-app.txt` (renamed to `requirements.txt` in the Space repo) to the Space's git remote
3. Required deps: `spaces`, `bitsandbytes`, `torch`, `transformers`, `peft`, `fastapi` — see `requirements-app.txt`

---

## Project Structure

```
sql-copilot/
├── train_lora.py           # LoRA fine-tuning (Colab or local GPU)
├── eval.py                 # Evaluation harness: fine-tuned vs base vs Groq
├── app.py                  # FastAPI + Gradio inference UI (deployed)
├── config.yaml             # Training config (Colab)
├── config.local.yaml       # Training config (local GPU, 4GB-tuned)
├── requirements.txt        # Local dev / eval deps
├── requirements-train.txt  # Colab/training deps
├── requirements-app.txt    # Deployment-only deps (pushed to the Space as requirements.txt)
└── README.md
```

---

## Limitations

1. **Low exact-match accuracy** on unseen schemas — a real, measured constraint of a lightly-trained small adapter on a strict string-match metric
2. **No execution-based accuracy** — the held-out dataset provides schemas but no populated databases, so correctness is judged by string match against gold SQL rather than by actually running the query
3. **ZeroGPU per-request reload** — every request pays a ~10-20s model-reload cost, not suitable for low-latency production use as-is
4. **Free-tier GPU quota** — the live demo can be temporarily rate-limited

## License

Apache 2.0 (matching Qwen2.5-Coder and the training dataset's licensing).
