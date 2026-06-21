# `results/` layout

The pipeline writes ALL persistent outputs under this directory. The
figures of the paper are generated outside this codebase — what's here
is the **raw, disaggregated data** that those figures consume.

## Top-level structure

```
results/
├── README.md                        ← this file (the schema)
├── diagnostico/                     ← base-model characterisation (no intervention)
│   └── <model-key>/                 ← e.g. llava-1.5-7b
│       └── <length>/                ← short | medium | long  (from prompt_key)
│           ├── <run-name>_LATEST.txt   ← pointer to the most recent run dir
│           └── <run-name>_<timestamp>/
│               ├── config_snapshot.yaml
│               ├── ref_captions.jsonl
│               ├── hidden_states/
│               │   ├── <image_id>__A.h5    ← real image
│               │   └── <image_id>__B.h5    ← noised image
│               ├── metrics.parquet         ← per-image rows (see "Schema" below)
│               ├── unit_cases/
│               │   ├── <image_id>.json     ← Fig-3 data per image
│               │   └── ... (one per image)
│               ├── summary.csv
│               ├── summary.json
│               ├── diagnostics.json
│               ├── diagnostics_collect.jsonl
│               └── logs/<script>.log
└── avaliacao/                       ← SPARC intervention vs baseline (paired)
    └── <model-key>/<length>/<run-name>_<timestamp>/
        ├── ... (mirrors diagnostico, with OFF + ON conditions)
        └── (wired in a later block — folder reserved for now)
```

`<length>` is derived from `cfg["task"]["prompt_key"]`
(`caption_{short,medium,long}`).
`<area>` is `cfg["run"]["area"]` (`"diagnostico"` by default;
`"avaliacao"` for SPARC runs).

## Schema — `metrics.parquet`

One row per image. Columns:

| column | type | notes |
|---|---|---|
| `image_id` | string | COCO id (zero-padded) |
| `caption_len` | int32 | length of caption_ref (tokens after caption_start) |
| `n_layers`, `hidden_dim` | int32 | model decoder depth + hidden width |
| `caption_ref` | string | the caption used as TF target |
| `kl` | `list<list<float32>>` | **Fig-1**: full KL matrix `(n_layers, caption_len)` |
| `cos_dist` | `list<list<float32>>` | **Fig-1**: full cosine-distance matrix |
| `deep_curve` | `list<float32>` (nullable) | **Fig-2**: deep-block-mean KL per token, length `caption_len` |
| `residual_ratio` | float32 | legacy (`mass(deep[t0:]) / mass(deep)`) |
| `share_tail` | float32 (nullable) | **post-Block-3 headline** (bounded `[0,1]`, SPARC-proof) |
| `head_tail_ratio` | float32 (nullable) | DEPRECATED (kept for back-compat) |
| `caption_tokens` | `list<string>` (nullable) | one token per position (drives Fig-3) |
| `model_id`, `prompt_key`, `seed_global`, `noise_seed`, `timestamp_iso` | identification |

## Schema — `unit_cases/<image_id>.json` (Fig 3)

```json
{
  "image_id": "000000000139",
  "image_path": "data/processed/mscoco_baseline/images/000000000139.jpg",
  "prompt": "Describe ...",
  "caption_ref": "A cozy living room ...",
  "caption_tokens": ["A", "cozy", ...],
  "caption_start": 612,
  "caption_len": 64,
  "share_tail": 0.42,
  "residual_ratio": 0.51,
  "metrics_parquet": "metrics.parquet",
  "kl_shape": [n_layers, caption_len],
  "cos_dist_shape": [n_layers, caption_len],
  "hallucinated_objects": [],
  "hallucinated_token_positions": [],
  "notes": "hallucinated_* are populated by a future CHAIR-on-diagnostic pass."
}
```

The `kl` and `cos_dist` matrices are NOT duplicated here — they live
once in `metrics.parquet`. The unit-case JSON points back to it so any
plotting code can `read_metrics_table(parquet_path)` and pick the row
with the matching `image_id`.

## Reading the data (Python)

```python
import pyarrow.parquet as pq
import numpy as np

rows = pq.read_table("results/diagnostico/llava-1.5-7b/long/<run>/metrics.parquet").to_pylist()
row = rows[0]
kl   = np.asarray(row["kl"],         dtype=np.float32)  # (n_layers, caption_len)
cos  = np.asarray(row["cos_dist"],   dtype=np.float32)
curve = np.asarray(row["deep_curve"], dtype=np.float32) # (caption_len,) — Fig 2
share = float(row["share_tail"])                        # Fig 2 / tables
```

For the unit case:

```python
import json
case = json.loads((run_dir / "unit_cases" / f"{image_id}.json").read_text())
# `case["metrics_parquet"]` → reopen the parquet, pull this image's row, plot.
```

## Hidden states (`hidden_states/<image_id>__{A,B}.h5`)

HDF5 files written by `io/storage.py`. Datasets:
- `hidden_states` — `(n_layers, seq_len, hidden_dim)`, float16
- `input_ids` — `(seq_len,)`, int64
- `attention_mask` — `(seq_len,)`, int8 (optional)

Attributes: `image_id`, `condition`, `caption_start`, `caption_len`,
`model_id`, `prompt_key`, `caption_ref`, `seed_global`, `noise_seed`,
`timestamp_iso`, `hidden_dim`.

These are not needed to remake the figures (the figures use derived
metrics from `metrics.parquet`) but are kept so any new metric can be
computed without re-running the model.

## Conventions

- Pointer file `<run-name>_LATEST.txt` at the `<area>/<model>/<length>/`
  level lets scripts find the most recent run without timestamp guessing.
- Timestamps are local `%Y-%m-%d_%H%M%S`.
- Persist raw + disaggregated. Means/medians/quartiles go in
  `summary.json` but the per-image rows in `metrics.parquet` are the
  source of truth — every figure must be derivable from them.
