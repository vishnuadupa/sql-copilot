# HuggingFace Space Setup (5 min)

This creates the live web UI for your model. **Do these steps once before Colab training.**

## Step 1: Create the Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `sql-copilot`
   - **License:** Apache 2.0
   - **SDK:** Docker
   - Click **Create Space**

3. You'll be redirected to: `https://huggingface.co/spaces/vishnuadupa/sql-copilot`

## Step 2: Clone the Space Repo

The Space is a git repo. Clone it:

```bash
git clone https://huggingface.co/spaces/vishnuadupa/sql-copilot space-repo
cd space-repo
```

## Step 3: Add Files to Space

Copy your project files into the Space repo:

```bash
# From your project directory
cd D:\ML\sql-copilot

# Copy key files to the Space
cp app.py ../space-repo/
cp Dockerfile ../space-repo/
cp requirements.txt ../space-repo/
cp config.yaml ../space-repo/
cp README.md ../space-repo/

cd ../space-repo
```

## Step 4: Create `.dockerignore` (optional but recommended)

```bash
echo "__pycache__/
*.pyc
.git
.gitignore
*.db
NEXT_STEPS.md
COLAB_TRAINING_GUIDE.md
train_lora.py
eval.py
.env" > .dockerignore
```

## Step 5: Commit & Push to HF Space

```bash
git add -A
git commit -m "Initial SQL Copilot deployment"
git push
```

## Step 6: Wait for Build

1. Go to https://huggingface.co/spaces/vishnuadupa/sql-copilot
2. Click "Build logs" in the top right
3. Wait for green checkmark (~5-10 min)
4. Once done, your Space URL is live!

## Step 7: Test

Visit: https://huggingface.co/spaces/vishnuadupa/sql-copilot

You'll see a Gradio UI where you can:
- Paste a natural language question
- Paste a database schema
- Click "Generate SQL" 
- See the generated SQL and execution results

---

## Important Notes

- **First run is slow:** The Space will download Qwen model (~3GB) on first load. Takes 2-3 min.
- **LoRA adapter:** App looks for `./qwen-sql-lora/` locally. After Colab training, you need to download the fine-tuned adapter and commit it to the Space repo, OR update `app.py` to load from HF Hub directly.

### Option A: Load from HF Hub (Recommended)

Edit `app.py` line ~16:

```python
# Before:
adapter_path = "./qwen-sql-lora"

# After:
adapter_path = "vishnuadupa/qwen-sql-lora"  # Load directly from HF Hub
```

Then commit:
```bash
git add app.py
git commit -m "Load model from HF Hub"
git push
```

This way, the Space always pulls your latest trained model automatically.

---

**Done!** Your Space is live. It'll wait for you to train the model and push the LoRA adapter to HF Hub.
