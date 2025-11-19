
# Amazon Product Pricer 

- **Data Source**: Amazon Reviews 2023 metadata (Appliances + multi-category blends) from Hugging Face.
- **Environment Setup**: create `.venv` in `week6/`, install `requirements.txt`, load secrets via `.env` (`HF_TOKEN`, optional `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). All notebooks expect `load_dotenv(override=True)` to populate these.
- **Notebooks**:
  - **day1.ipynb** – explored appliance metadata, validated the `Item` prompt builder, and plotted price/token distributions.  
  - **day2.ipynb** – multi-category curation: balanced price slots, assigned categories, saved 400k-train / 2k-test sets (`train.pkl`, `test.pkl`), and prepped histograms.  
  - **lite.ipynb** – appliance-only “lite” pipeline producing `train_lite.pkl` / `test_lite.pkl` for faster experimentation.  
  - **day3.ipynb** – loaded pickled items and benchmarked classic ML baselines (random guess, constant average, linear regression, bag-of-words LR, Word2Vec+LR, Linear SVR, Random Forest) using the shared `Tester` harness to report average error, RMSLE, and green/orange/red hit rates.

- **APIs & Tokens Needed**:
  - `HF_TOKEN` (read, optional write) to authenticate dataset downloads and pushes.
  - `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` only required later when comparing frontier models (Day 4+) or fine-tuning (Day 5). Safe to omit if you’re staying on Day 1–3 flows.

- **Pipeline Artifacts**:
  - `train.pkl`, `test.pkl` – full multi-category items (`Item` instances with prompts).
  - `train_lite.pkl`, `test_lite.pkl` – reduced appliance-only version.
  - Optional: `human_input.csv`, `human_output.csv` for manual benchmark collection (Day 4 prep).

- **Baseline Results Snapshot (Day 3)**:
  - `random_pricer` – wide error (hundreds of dollars, RMSLE highest).
  - `constant_pricer` (global average) – smaller but still poor; baseline floor.
  - `linear_regression_pricer` with hand-crafted numeric features – modest improvement.
  - `bow_lr_pricer` (CountVectorizer + linear regression) – better RMSLE thanks to richer text features.
  - `word2vec_lr_pricer` – further improvement via dense embeddings.
  - `svr_pricer` – similar to Word2Vec LR, sometimes more robust on outliers.
  - `random_forest_pricer` – best-performing baseline in this notebook; lowest RMSLE/higher green hit rate among classic models.

- **Common Issues Encountered**:
  - Invalid or missing Hugging Face tokens (401 errors) and gated model access (403 for Meta Llama 3.1).
  - Hugging Face dataset downloads taking ~1h on first run; subsequent runs hit the on-disk cache.
  - Transformers warning about missing PyTorch—installing `torch==2.4.1` resolves it when local inference/fine-tuning is required.

- **Next Steps**:
  - Continue into Day 4 (frontier model benchmarks) and Day 5 (fine-tuning).
  - Begin modularizing notebooks into scripts/CLI for the production pipeline plan (Phase 1 onward).
