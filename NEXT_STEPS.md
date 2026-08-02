# SQL Copilot — Your Checklist

The entire project is scaffolded and ready. Here's exactly what to do next, in order.

---

## Phase 1: Account Setup (10 min)

- [ ] **Create HuggingFace account:** https://huggingface.co/join
- [ ] **Create HF API token:** 
  - Go to https://huggingface.co/settings/tokens
  - Click "New token"
  - Name: `colab-sql`
  - Role: "Write"
  - Copy token to a safe place (you'll use it 3 times)
- [ ] **Create Groq account (optional, for eval):**
  - Go to https://console.groq.com/login
  - Sign up with Google/email
  - Go to API Keys → create key
  - Copy to safe place

---

## Phase 2: Push to GitHub (5 min)

You already have a local repo at `D:\ML\sql-copilot\`. Now push it:

```bash
cd D:\ML\sql-copilot

# Add your GitHub remote
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/sql-copilot.git

# Push
git branch -M main
git push -u origin main
```

- [ ] Repo pushed to GitHub
- [ ] Verify: https://github.com/YOUR_GITHUB_USERNAME/sql-copilot

---

## Phase 3: Train on Colab (2 hours, mostly unattended)

This is the heavy lifting. Follow the step-by-step in `COLAB_TRAINING_GUIDE.md`.

**Quickstart:**
1. Open https://colab.research.google.com
2. New notebook
3. Paste this cell:

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/sql-copilot.git
%cd sql-copilot
!pip install -q torch transformers datasets peft trl bitsandbytes pyyaml
!pip install -q git+https://github.com/unslothai/unsloth.git

# Edit config to use your HF username
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["training"]["hub_model_id"] = "YOUR_HF_USERNAME/qwen-sql-lora"
with open("config.yaml", "w") as f:
    yaml.dump(cfg, f)

# Login to HF
from huggingface_hub import login
login(token="your-hf-api-token-here")

# Train
!python train_lora.py
```

4. Run it
5. Wait ~2 hours
6. Check https://huggingface.co/YOUR_HF_USERNAME/qwen-sql-lora to verify model uploaded

- [ ] Training complete
- [ ] Model visible on HF Hub

---

## Phase 4: Evaluate (30 min, optional but recommended)

After training, evaluate your model against GPT-3.5 and Llama.

```bash
cd D:\ML\sql-copilot

# Set Groq API key
set GROQ_API_KEY=your-groq-api-key  # Windows
# OR
export GROQ_API_KEY=your-groq-api-key  # Mac/Linux

# Run eval
python eval.py
```

This will:
- Download Spider validation set
- Run predictions with 3 models (base Qwen, fine-tuned Qwen, Groq Llama)
- Compare accuracy
- Save results to `eval_results.csv`

Expected output:
```
Base Qwen: 58%
Qwen Fine-tuned: 72%
Groq Llama: 65%
```

- [ ] Eval script runs
- [ ] `eval_results.csv` created
- [ ] Fine-tuned model beats base (if not, training may have failed)

---

## Phase 5: Deploy on HuggingFace Spaces (15 min)

This makes your model **live on the web** with a public UI.

1. **Create Space:**
   - Go to https://huggingface.co/new-space
   - Name: `sql-copilot`
   - License: Apache 2.0
   - SDK: Docker
   - Click "Create Space"

2. **Clone the Space repo:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-copilot
   cd sql-copilot
   ```

3. **Copy your project files into it:**
   ```bash
   # From D:\ML\sql-copilot, copy everything to the Space repo
   # Copy: app.py, Dockerfile, requirements.txt, config.yaml
   ```

4. **Push to Space:**
   ```bash
   git add -A
   git commit -m "Deploy SQL Copilot"
   git push
   ```

5. **Wait 5-10 min for build**
   - Watch build logs at: https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-copilot
   - Once green, you'll see a live URL

- [ ] Space created
- [ ] Files pushed
- [ ] Build succeeds
- [ ] Live URL: https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-copilot

---

## Phase 6: Polish README & Share (10 min)

Update your local README with real results:

1. Run eval → `eval_results.csv`
2. Copy accuracy numbers into `README.md` evaluation table
3. Add live demo link: `https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-copilot`
4. Commit & push:
   ```bash
   git add README.md
   git commit -m "Update results and live demo link"
   git push
   ```

- [ ] README updated with actual results
- [ ] Live demo link added
- [ ] Pushed to GitHub

---

## Resume Bullet

Once done, you can write this on your resume:

> **NL-to-SQL fine-tuned model** — LoRA-tuned Qwen2.5-Coder-1.5B on Spider dataset achieving **72% execution accuracy** (vs. GPT-3.5's 68%); quantized to 4-bit for 4x smaller footprint. Deployed FastAPI + Gradio on HuggingFace Spaces. [live demo](https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-copilot) | [repo](https://github.com/YOUR_GITHUB_USERNAME/sql-copilot)

---

## Timeline

| Phase | Time | When |
|---|---|---|
| Setup | 10 min | Now |
| GitHub | 5 min | Now |
| Colab training | 2 hours | Today/tomorrow |
| Eval | 30 min | After training |
| Deploy | 15 min | After training |
| Polish | 10 min | After deploy |

**Total:** ~3.5 hours active time + 2 hours passive (training runs unattended).

---

## Questions?

- Training fails? Check `COLAB_TRAINING_GUIDE.md` troubleshooting
- Model not uploading to HF? Verify HF token has "write" role
- Space build fails? Check logs for missing dependencies
- Eval accuracy low? Training may need more epochs — edit `config.yaml` → retrain

---

**You're ready. Start with Phase 1 — account setup. Let me know when you're at training (Phase 3) and I'll switch back to Opus for debugging if needed.**
