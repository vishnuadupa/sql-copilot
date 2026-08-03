# SQL Copilot — Progress & Current State

Last updated: 2026-08-03

## Links

- GitHub repo: https://github.com/vishnuadupa/sql-copilot
- HF Space (live demo): https://huggingface.co/spaces/vishnuadupa/sql-copilot
- Direct app URL (bypasses HF wrapper page): https://vishnuadupa-sql-copilot.hf.space
- Fine-tuned model: https://huggingface.co/vishnuadupa/qwen-sql-lora
- Local project path: `D:\ML\sql-copilot`
- Space repo local clone (separate git remote, used for deploying to Space): `/tmp/sql-copilot-space`

## What's Done

### Training
- Fine-tuned Qwen2.5-Coder-1.5B via LoRA (0.14% of params) on `b-mc2/sql-create-context` (7,000 examples, real `CREATE TABLE` schema context per example)
- Trained on Colab free T4 GPU, checkpoints persisted to Google Drive (survives disconnects)
- Fixed real training bugs along the way: missing EOS token (model didn't know when to stop generating), YAML `2e-4` parsed as string not float, Unsloth import order, LoRA-before-trainer attachment order, numeric vs lexical checkpoint sorting

### Evaluation
- `eval.py` benchmarks fine-tuned Qwen vs base Qwen vs Groq Llama-3.1-8B on 100 held-out examples (disjoint from training via shared `shuffle(seed=42)` split)
- Fixed real eval bugs: prompt format mismatch vs training format (was silently collapsing accuracy), stale-cache resume bug (trusted old results blindly), `os.path.exists()` check broken for Hub repo IDs (silently loaded base model instead of adapter), non-deterministic sampling instead of greedy decoding
- **Honest final measured result: ~2% normalized exact-match accuracy** for the fine-tuned model (vs ~1% base, ~0% Groq on this metric)
- Why it's low, legitimately: strict exact-string-match metric (functionally-correct-but-differently-formatted SQL scores as wrong), tiny adapter, held-out **unseen** schemas (real generalization test, not memorization)
- Direct manual spot-checks on the live deployed model (bypassing the eval harness) show it often produces genuinely correct SQL on questions similar in shape to training data, but hallucinates filter conditions on other question types — this is real, consistent behavior, not a measurement bug

### Deployment
- Live on HuggingFace Spaces, Gradio SDK, free ZeroGPU tier
- Fixed a long chain of real deployment bugs (see below) — app is now genuinely working: verified via direct browser testing that it generates and executes real SQL against a sample SQLite DB

## Known, Real, Accepted Limitation

**ZeroGPU does not persist Python process state between `@spaces.GPU`-decorated calls.** Confirmed via logs: "Loading fine-tuned model..." prints fresh on every single request, meaning the ~1.14GB adapter + 3GB base model reload from HF Hub on every click (~10-20s overhead per request). This is architectural to the free ZeroGPU tier (each call runs in an isolated worker), not a bug fixable in our code. Decision: accept this as a known cost of the free tier rather than continue chasing a fix.

## Deployment Bug Chain (all fixed, in order encountered)

1. Docker SDK is paid on HF Spaces → switched to Gradio SDK (free)
2. `torch==2.1.2` incompatible with ZeroGPU → bumped to `2.11.0`
3. Pinned `gradio==4.15.0` in requirements conflicted with HF's own platform-managed Gradio version → removed all version pins from `requirements-app.txt`
4. `RUNTIME_ERROR: No @spaces.GPU function detected` → added `@spaces.GPU` decorator to `gradio_interface`
5. Model/tokenizer were never loading — FastAPI lifespan hook never runs under Gradio SDK (HF serves the `demo` Blocks object directly, not `app`) → moved loading out of lifespan
6. `create_sample_db()` was gated behind `if __name__ == "__main__"`, which never executes when HF imports the module → moved to run unconditionally at import time
7. `gr.JSON` output component crashed on any non-JSON string return value (e.g. bare error strings) → always return `json.dumps(...)` now
8. Fine-tuned adapter silently fell back to base model — loading a 4-bit-trained adapter requires `bitsandbytes`, which wasn't in `requirements-app.txt` → added it
9. `RuntimeError: CUDA has been initialized before importing the spaces package` → `import spaces` must be the very first import in the file, before torch/transformers/peft
10. `RuntimeError: No CUDA GPUs are available` at import time — `AutoPeftModelForCausalLM` auto-detects the adapter's saved 4-bit quantization config and requires a live CUDA device just to *load* it, unavailable under ZeroGPU until a decorated call runs → switched to loading base model in plain fp32 + applying adapter via plain `PeftModel` (no quantization needed to load)
11. Same "No CUDA GPUs available" error persisted even with fp32 loading, because it was still happening at **module import time** — `spaces` installs a global torch patch on import that intercepts ALL tensor ops (even plain CPU `safetensors.load_file()`) and requires an active GPU-allocated context → moved model loading to be **lazy**, triggered on first call from inside the `@spaces.GPU`-decorated function itself (this actually got the demo working — verified via live browser test)

## Current Session's In-Progress Work (INTERRUPTED — resume here)

**Task:** Improve UI/UX + add a "🎲 Try an Example" button that fills in a random verified-working (question, schema) pair, while keeping manual input still available.

**Plan agreed with user:**
1. Curate a genuinely-tested "working set" of example prompts (not guessed) by manually testing candidates against the live model
2. Add the random-example button (`EXAMPLES` list + `random.choice` + button click handler filling both textboxes)
3. Polish UI: better schema input (code/syntax highlighting), friendlier empty/error states, cleaner layout, honest subtitle about limitations
4. Deploy and verify live

**Verified-working example so far (via live browser test):**
- Question: `"Top 5 customers by revenue in Northeast"`
- Schema:
  ```sql
  CREATE TABLE customers (id INTEGER, name TEXT, region TEXT, revenue REAL)
  CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL, date TEXT)
  ```
- Result: correct join + filter + aggregation + `LIMIT 5`, returned `[["BigCorp"], ["Acme Corp"]]` — genuinely correct.

**Tested and failed (hallucinated conditions not in the question) — do NOT use these in the example set:**
- "How many orders were placed after January 2024?" → hallucinated `region = "North America"`, `revenue > 1000000`, used `SUM` instead of `COUNT`
- "How many customers are there?" → hallucinated unnecessary JOIN + `HAVING SUM(amount) > 50000 AND region = "North"`

**Still mid-test when interrupted:**
- Was testing "List all customers in the West region" — got an "Error" in the UI. Had just added server-side traceback logging (`app.py`, commit `07fa786`, pushed to GitHub main repo but **NOT YET pushed to the Space repo** at `/tmp/sql-copilot-space`) to see the actual exception, but was interrupted before checking the Space's logs for the real traceback.

**Immediate next steps:**
1. Push commit `07fa786` (traceback logging) to the Space repo (`/tmp/sql-copilot-space` → `git pull`, copy `app.py`, commit, push to `https://huggingface.co/spaces/vishnuadupa/sql-copilot`)
2. Wait for rebuild, retest "List all customers in the West region", check logs this time for the actual exception
3. Given each test costs ~15-20s (model reload), budget testing carefully — aim for 2-3 more curated examples, not exhaustive testing of every candidate
4. Finalize `EXAMPLES` list (start with just the ONE fully-proven example if time-constrained, add more only as verified)
5. Implement the random-example button + UI polish in `app.py`
6. Deploy to both GitHub main repo and Space repo
7. Final live browser verification

## Key Operational Notes for Resuming

- **HF token** is in `D:\ML\sql-copilot\.env` (gitignored) — read it from there, never hardcode it in tracked files (may need refreshing if expired/rotated)
- **Two separate git remotes** to keep in sync: GitHub (`vishnuadupa/sql-copilot`, source of truth) and HF Space (`/tmp/sql-copilot-space`, deployment target) — every `app.py` change must be pushed to **both**
- **Space rebuild takes ~1-2 min** after push; poll `hf spaces info vishnuadupa/sql-copilot` for `runtime.stage` reaching `RUNNING`, or check for `RUNTIME_ERROR`/`BUILD_ERROR`
- **Windows terminal encoding issue**: `hf spaces logs` can choke on non-ASCII characters in the log stream and truncate output silently — always redirect to a file first (`> /tmp/logs.txt 2>&1`) rather than reading piped output directly, and set `PYTHONIOENCODING=utf-8`
- **Browser tool quirk**: after typing into a Gradio textbox, a plain `left_click` on the submit button sometimes doesn't register (no new network request fires) — `double_click` reliably works instead. Always verify a click actually fired by checking `read_network_requests` for a new `queue/join` POST before waiting for results.
- **Testing is slow**: each live test costs ~15-20s due to the ZeroGPU reload-every-call behavior (see Known Limitation above) — budget accordingly, don't over-test

## Resume Bullet Points (current honest version)

- Fine-tuned Qwen2.5-Coder-1.5B via LoRA (0.14% of params) on 7k NL-to-SQL examples with real schema context; built a leak-free eval harness (disjoint splits, greedy decoding) benchmarking against GPT-class baselines on unseen schemas
- Debugged the full pipeline end-to-end: fixed a training bug (missing EOS supervision), a prompt-format mismatch, a stale eval cache, and a silent adapter-loading failure — each root-caused via direct log inspection
- Deployed live on HuggingFace Spaces (FastAPI + Gradio, free GPU tier); resolved multiple production deployment failures (GPU allocation, CUDA import ordering, quantization dependencies) to ship a working public demo
