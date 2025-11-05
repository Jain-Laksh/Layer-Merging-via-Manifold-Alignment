Data-Dependent Layer Merging
============================

Overview
--------
- Compute layer-wise similarity matrices for large language models using diffusion kernel embeddings and a Gaussian mutual information score.
- The provided scripts target the MMLU benchmark and can be adapted to other datasets with minimal changes.

Environment
-----------
- Python 3.10 or later.
- Install dependencies with:

  ```
  pip install torch transformers datasets pandas numpy scikit-learn
  ```

- Access to the Meta Llama 3 family (or another causal LM) via Hugging Face is required; ensure you have accepted the model license and configured credentials (`huggingface-cli login` or environment tokens).

Repository Layout
-----------------
- `scripts/compute_similarity.py` — loads a causal LM, extracts hidden states per transformer layer, applies diffusion embeddings, and saves the similarity matrix.
- `scripts/download_mmlu.py` — fetches the MMLU test CSVs into `~/aryan/data/mmlu/test`.
- `scripts/utils_dataset.py` — utilities to sample grouped MMLU tasks (medical, legal, math, cs, humanities).
- `scripts/utils_merging.py` — diffusion kernel embedding and similarity computation helpers.
- `notebooks/mmlu.ipynb` — exploratory notebook for inspecting the generated similarity matrices.
- `outputs/mmlu/` — example similarity matrices generated with 40 samples per task.
- `model/model.txt` — reminder with the canonical download location for Meta Llama 3 8B weights.

Setup
-----
1. **Model weights** — Download `meta-llama/Meta-Llama-3-8B` (or another supported causal LM) into a local directory, e.g. `~/aryan/models/llama3-8b`.
2. **MMLU data** — Run:

   ```
   python scripts/download_mmlu.py
   ```

   The script saves per-subject CSVs to `~/aryan/data/mmlu/test`, matching the default path expected by `compute_similarity.py`.
3. **GPU configuration** — The similarity script prefers a GPU with bfloat16 or float16 support. If you only have CPU access, adjust `device_map` and dtype in `compute_similarity.py` accordingly (runtime will be significantly slower).

Computing Layer Similarity
--------------------------
- The main entry point is `scripts/compute_similarity.py`:

  ```
  python scripts/compute_similarity.py \
    --model_path ~/aryan/models/llama3-8b \
    --data_dir ~/aryan/data/mmlu \
    --task_name medical \
    --num_samples 40 \
    --output_dir ~/aryan/outputs/similarity_mats
  ```

- Supported task groups: `medical`, `legal`, `math`, `cs`, `humanities`. Each group samples across two related MMLU subjects (see `utils_dataset.py`).
- `--num_samples` controls how many prompts are sampled per task (defaults to 40). Increase for a more stable similarity estimate at the cost of compute and memory.
- The command can be batched over several tasks. For example:

  ```
  for task in medical legal math cs humanities; do
    python scripts/compute_similarity.py \
      --model_path ~/aryan/models/llama3-8b \
      --data_dir ~/aryan/data/mmlu \
      --task_name $task \
      --num_samples 40 \
      --output_dir ~/aryan/outputs/similarity_mats
  done
  ```

Multilingual Global-MMLU
------------------------
- For bilingual/multilingual evaluation using CohereLabs' `Global-MMLU` translations, use the helper script `aryan/run_multilingual_similarity.py` (relative to the project root).
- Example command (translates the medical task into Simplified Chinese, Spanish, and French while keeping English prompts) on a Studio workspace path:

  ```
  python3 aryan/run_multilingual_similarity.py \
    --model_path /teamspace/studios/this_studio/aryan/models/llama3-8b \
    --task_name medical \
    --num_samples 20 \
    --global_languages zh es fr \
    --device_map auto
  ```

- Ensure the `CohereLabs/Global-MMLU` dataset is available locally or cached via `datasets` so the script can load the language-specific splits.
