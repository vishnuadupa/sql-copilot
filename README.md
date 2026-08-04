# SQL Copilot: Fine-Tuned NL → SQL with Qwen2.5-Coder

**A small (1.5B param) LoRA-fine-tuned model that translates natural language questions into SQL, deployed live on HuggingFace Spaces.**

🔗 [Live Demo](https://huggingface.co/spaces/vishnuadupa/sql-copilot) · [Fine-tuned Model](https://huggingface.co/vishnuadupa/qwen-sql-lora) · [Progress Log](PROGRESS.md)

---

## What this actually is

This project fine-tunes Qwen2.5-Coder-1.5B via LoRA to generate SQL from a natural-language question + a database schema, then deploys it as a live, interactive demo. It also includes a leak-free evaluation harness comparing the fine-tuned model against the untrained base model and a Groq-hosted Llama-3.1-8B baseline.

**The honest headline number:** ~2% normalized exact-match accuracy on 100 held-out, unseen-schema questions. That's low, and this README explains exactly why, rather than hiding it — see [Results](#results) and [Limitations](#limitations) below.

---

## Results

Measured on 100 examples held out from training (disjoint via a shared `shuffle(seed=42)` split — see `train_lora.py` / `eval.py`), using greedy decoding and a prompt format identical to training:

| Model | Normalized exact-match accuracy |
|---|---|
| Qwen2.5-Coder-1.5B (fine-tuned) | ~2% |
| Qwen2.5-Coder-1.5B (base, no fine-tuning) | ~1% |
| Llama-3.1-8B (Groq, zero-shot) | ~0% |

### Why this number is low, and why it's still a legitimate result

1. **Strict metric.** Exact-string-match fails a query that is functionally identical but differently formatted — reordered `WHERE` clauses, different quoting, a missing/extra `SUM()` are all scored as wrong even when the SQL logic is fine. Manual spot-checks (see `PROGRESS.md`) showed the model getting several questions exactly right, and others producing SQL that's *close* but not byte-identical to the gold answer.
2. **Tiny adapter.** Only 0.14% of the model's parameters were trained (LoRA rank 16 on 2 projection matrices).
3. **Genuine held-out generalization.** The evaluation set uses table/column names the model never saw during training — this is a real test of generalization, not memorization.
4. **Live behavior is consistent with this.** Testing the deployed model directly shows it sometimes generates exactly correct joins/filters/aggregates, and sometimes hallucinates a filter condition that was never asked for. Both are shown deliberately in the live demo's UI (see the "Honest note" in the app itself).

**The stronger story here isn't the accuracy number — it's the engineering.** Building this end-to-end (data → training → evaluation → deployment) surfaced and required fixing over a dozen real, independently-diagnosed bugs across the whole pipeline — see [Bug History](#notable-bugs-fixed) below.

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
- **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — includes real `CREATE TABLE` schema text per example (not just a bare database name)
- **Split:** `full.shuffle(seed=42)`; first 100 examples → eval set, next 7,000 → training set. Same seed used in both `train_lora.py` and `eval.py`, guaranteeing zero overlap.

### Fine-tuning
- **Base model:** Qwen2.5-Coder-1.5B-Instruct (Apache 2.0)
- **Method:** LoRA, rank 16, alpha 32, targeting `q_proj`/`v_proj`, dropout 0.05 (2,179,072 trainable params — 0.14% of the model)
- **Training text includes an explicit EOS token** after each SQL answer — without this, the model never learns where an answer ends and appends extra, wrong clauses after a correct query (a real bug found and fixed; see bug history)
- **Hardware:** Colab free T4 (~35 min for 3 epochs / 2,625 steps), with checkpoints on Google Drive so a Colab disconnect never loses progress
- **Config:** see `config.yaml` (Colab) / `config.local.yaml` (local GPU training, tuned for 4GB VRAM)

### Evaluation
- **Metric:** normalized exact-match (lowercased, whitespace-collapsed, trailing semicolon stripped)
- **Decoding:** greedy (`do_sample=False`) — sampling was tried first and produced genuine run-to-run score variance on identical inputs, which isn't valid for an accuracy measurement
- **Resume-safe:** results save incrementally and are tagged with a model+adapter signature, so a stale/incompatible results file is never silently trusted as "already complete"

### Deployment
- **Stack:** FastAPI (`/predict` API) + Gradio (interactive UI), served together from `app.py`
- **Hosting:** HuggingFace Spaces, free ZeroGPU tier
- **Known limitation:** ZeroGPU does not persist Python process state between GPU-decorated calls — the model reloads from HF Hub on every single request (~10-20s overhead). This is architectural to the free tier, not fixable in application code; accepted as a known cost.
- **Free tier also has a daily GPU-time quota** — the demo may show a GPU-acquisition error if the quota is exhausted; it resets on HF's schedule.

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
`config.yaml`'s `hub_model_id` already points at a real, pushed model — change it to your own HF username before training your own version.

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
1. Create a Space at https://huggingface.co/new-space with **SDK: Gradio** (not Docker — Docker is a paid-tier feature; the `Dockerfile` in this repo is unused/historical from an earlier deployment attempt)
2. Push `app.py` + `requirements-app.txt` (renamed to `requirements.txt` in the Space repo) to the Space's git remote
3. The Space needs `spaces`, `bitsandbytes`, `torch`, `transformers`, `peft`, `fastapi` — see `requirements-app.txt`

---

## Project Structure

```
sql-copilot/
├── train_lora.py           # LoRA fine-tuning (Colab or local GPU)
├── eval.py                 # Evaluation harness: fine-tuned vs base vs Groq
├── app.py                  # FastAPI + Gradio inference UI (deployed)
├── config.yaml              # Training config (Colab)
├── config.local.yaml        # Training config (local GPU, 4GB-tuned)
├── requirements.txt         # Local dev / eval deps
├── requirements-train.txt   # Colab/training deps (no version pins — unsloth manages these)
├── requirements-app.txt     # Deployment-only deps (pushed to the Space as requirements.txt)
├── Dockerfile               # Unused — kept from an earlier Docker-based deployment attempt
├── PROGRESS.md              # Detailed session log: bugs found, fixes, current state
└── README.md                 # This file
```

---

## Notable Bugs Fixed

A representative sample of real, independently-diagnosed issues hit while building this (full list in `PROGRESS.md`):

- **Training never learned to stop** — no EOS token in training text meant the model appended extra, wrong SQL clauses after a correct answer
- **Eval prompt didn't match training format** — different wording + a missing trailing newline collapsed measured accuracy from ~60% (on a hand-tested example) to ~2%
- **Silent adapter-loading failure** — `os.path.exists()` doesn't work for HF Hub repo IDs, so the "fine-tuned" eval column was silently running the untrained base model
- **Stale eval cache** — a resume feature trusted any pre-existing results file as "already complete" regardless of which model produced it
- **Non-deterministic eval** — random sampling (`temperature=0.7`) gave different scores across runs on identical inputs; switched to greedy decoding
- **ZeroGPU deployment chain** (~6 distinct errors): missing `@spaces.GPU` decorator, FastAPI lifespan never running under Gradio SDK, `bitsandbytes` missing from deploy deps, `spaces` import-order requirement, 4-bit quantization auto-detection requiring a GPU at import time, and finally — model loading needing to happen *inside* the GPU-decorated call rather than at module import, since ZeroGPU grants no GPU context until then

---

## Limitations

1. **Low exact-match accuracy** on unseen schemas (see Results) — this is a real, measured limitation, not a placeholder
2. **No true "execution accuracy"** metric — the held-out dataset provides schemas but no populated databases, so correctness is judged by string match against gold SQL, not by running the query
3. **ZeroGPU per-request reload** — every request pays a ~10-20s model-reload cost; not suitable for low-latency production use as-is
4. **Free-tier GPU quota** — the live demo can be temporarily rate-limited

## License

Apache 2.0 (matching Qwen2.5-Coder and the training dataset's licensing).
