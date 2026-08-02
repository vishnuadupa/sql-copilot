# HuggingFace Space Setup (5 min)

This creates the live web UI for your model. **Do these steps once before Colab training.**

## Step 1: Create the Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `sql-copilot`
   - **License:** Apache 2.0
   - **SDK:** Gradio (FREE)
   - Click **Create Space**

3. You'll be redirected to: `https://huggingface.co/spaces/vishnuadupa/sql-copilot`

## Step 2: Clone the Space Repo

The Space is a git repo. Clone it:

```bash
git clone https://huggingface.co/spaces/vishnuadupa/sql-copilot space-repo
cd space-repo
```

## Step 3: Add Required Files

Copy minimal files to the Space repo:

```bash
# From your project directory
cd D:\ML\sql-copilot

# Copy to Space
cp app.py ../space-repo/
cp requirements.txt ../space-repo/

cd ../space-repo
```

## Step 4: Create `.gitignore`

```bash
echo "__pycache__/
*.pyc
*.db
*.sqlite
.env
runs/
qwen-sql-lora/" > .gitignore
```

## Step 5: Commit & Push to HF Space

```bash
git add -A
git commit -m "Initial SQL Copilot deployment with Gradio"
git push
```

## Step 6: Wait for Build

1. Go to https://huggingface.co/spaces/vishnuadupa/sql-copilot
2. Space auto-builds (should be <1 min for Gradio, no Docker overhead)
3. Once green, your Space URL is live!

## Step 7: Test

Visit: https://huggingface.co/spaces/vishnuadupa/sql-copilot

You'll see a Gradio UI where you can:
- Paste a natural language question
- Paste a database schema
- Click "Generate SQL" 
- See the generated SQL and execution results

---

## Important Notes

- **Gradio SDK is free** — no Docker overhead, instant deployment
- **First run is slow:** The Space will download Qwen model (~3GB) on first load. Takes 2-3 min.
- **Model loads from HF Hub:** `app.py` is already configured to load your fine-tuned adapter from HF Hub automatically. Once you train on Colab, the Space automatically uses your latest model. ✅

---

**Done!** Your Space is live and configured. After Colab training finishes, the Space will automatically use your new fine-tuned model. No manual updates needed.
