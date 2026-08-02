# Colab Training Guide — SQL Copilot

This guide walks you through training on Google Colab (free T4 GPU, ~2 hours).

## Step 0: Prerequisites (10 min)

### Create HuggingFace Account
1. Go to https://huggingface.co/join
2. Sign up with email
3. Go to https://huggingface.co/settings/tokens
4. Click "New token"
5. Name: "colab-sql"
6. Role: "Write"
7. Copy token (you'll paste it in Colab)

### Create Groq Account (optional, for evaluation)
1. Go to https://console.groq.com
2. Sign up with Google/email
3. Go to API Keys
4. Create new key, copy it

## Step 1: Open Colab

1. Go to https://colab.research.google.com
2. Click "New notebook"
3. Rename to "SQL Copilot Training"

## Step 2: Clone Repo & Install

In first cell, paste:

```python
# Clone repo
!git clone https://github.com/YOUR_GITHUB_USERNAME/sql-copilot.git
%cd sql-copilot

# Check GPU
!nvidia-smi
```

Run it (Ctrl+Enter). You should see Tesla T4 or P100.

## Step 3: Update Config

In next cell:

```python
# Edit config.yaml with your HF username
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["hub_model_id"] = "YOUR_HF_USERNAME/qwen-sql-lora"

with open("config.yaml", "w") as f:
    yaml.dump(cfg, f)

print(f"Updated config: {cfg['training']['hub_model_id']}")
```

Replace `YOUR_HF_USERNAME` with your actual HuggingFace username.

## Step 4: Login to HuggingFace

```python
from huggingface_hub import login

login(token="paste_your_hf_token_here")
```

Paste your HF API token (from Step 0).

## Step 5: Install Dependencies

```python
!pip install -q torch transformers datasets peft unsloth trl bitsandbytes pyyaml

# Unsloth is special (much faster LoRA training)
!pip install -q git+https://github.com/unslothai/unsloth.git
```

Wait ~3 minutes for deps to install.

## Step 6: Run Training

```python
!python train_lora.py
```

This will:
1. Load Spider dataset (auto-downloads, ~500MB)
2. Load Qwen2.5-Coder-1.5B in 4-bit
3. Fine-tune with LoRA for 3 epochs
4. Save checkpoints locally
5. Push adapter to HF Hub every 200 steps

**Expected time: 1.5 - 2 hours**

Metrics to watch:
- `train_loss` should decrease from ~2.0 → ~0.5
- `eval_loss` every 200 steps
- Check your HF Hub repo for auto-pushed checkpoints

## Step 7: Verify Model Uploaded

After training completes:

```python
# List your models
from huggingface_hub import list_repo_files

try:
    files = list_repo_files("YOUR_HF_USERNAME/qwen-sql-lora", repo_type="model")
    print("✓ Model uploaded successfully!")
    print(f"Files: {files}")
except:
    print("✗ Model not found. Check HF Hub.")
```

## Step 8: Download & Test Locally (optional)

If you want to test the fine-tuned model in Colab before deploying:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

adapter_path = "YOUR_HF_USERNAME/qwen-sql-lora"
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto")

# Test
prompt = """You are a SQL expert. Generate SQL only.

Question: Top 5 customers by revenue in Northeast

Schema:
Table: customers (id, name, region, revenue)
Table: orders (id, customer_id, amount, date)

SQL:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `RuntimeError: out of memory` | Reduce `batch_size` in config.yaml to 2 |
| `HF login failed` | Regenerate token at https://huggingface.co/settings/tokens |
| `Spider dataset not found` | Run `from datasets import load_dataset; load_dataset("yale-lily/spider")` to cache it |
| `Unsloth install fails` | Skip it; just use regular transformers (slower but works) |

---

## Next Steps

1. After training finishes, clone your repo locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sql-copilot.git
   cd sql-copilot
   ```

2. Run evaluation (requires GROQ_API_KEY):
   ```bash
   export GROQ_API_KEY="your_groq_key"
   python eval.py
   ```

3. Deploy to HF Space (see README.md)

---

**Done!** Your fine-tuned model is live on HF Hub. Next: evaluate and deploy.
