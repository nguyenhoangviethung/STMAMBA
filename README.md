# NExT-ST-Mamba

Research codebase for NExT-ST-Mamba (spatio-temporal VideoQA research).

Quickstart
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate a tiny sample dataset for quick experiments (creates `datasets/sample`):

```bash
python3 scripts/create_sample_data.py --out datasets/sample
```

3. Run a lightweight Phase 3 forward test (no heavy SLM downloads):

```bash
python3 scripts/test_phase3.py
```

4. Train on sample data (very small):

```bash
python3 scripts/train.py --train-csv datasets/sample/train.csv --val-csv datasets/sample/val.csv --feat-h5-train datasets/sample/app_mot_train.h5 --feat-h5-val datasets/sample/app_mot_val.h5 --epochs 1 --batch-size 2
```

Notes
- The SLM reasoner (`QwenReasoner`) is optional and heavy; by default training avoids downloading it. To enable generation, initialize `NextSTMamba` with `load_reasoner=True`.
- HDF5 feature loading uses `h5py` and opens files lazily per worker to avoid repeated open/close overhead.
- GRPO is implemented as a memory-efficient surrogate that reweights supervised losses by group-relative advantages. For true policy gradients, add log-prob extraction to the reasoner.

Development
- Run unit tests (when available): `pytest`
- Linting: `flake8` (not preconfigured)

Files of interest
- `data/nextqa_dataset.py`: HDF5-backed dataset and ActionChunk transforms
- `models/next_st_mamba.py`: model assembly and generation bridge
- `training/lightning_module.py`: Lightning training loop using GRPO
- `scripts/train.py` and `scripts/evaluate.py`: entrypoints

If you'd like, I can add CI, README badges, or prepare a Colab notebook for quick demo.
