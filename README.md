# learning-from-preferences

Research code for **reinforcement learning from human preference feedback**: an agent learns a
reward model from pairwise comparisons of trajectory segments ("which of these two clips is
better?") and then optimises a policy against that learned reward.

This repository groups preference-based algorithms by algorithm. Each one lives in its own
self-contained folder under `algorithms/` with its own dependencies, launch scripts and upstream
license.

## Layout

```
learning-from-preferences/
├── README.md
├── .gitignore
└── algorithms/
    ├── preference_transformer/   # Preference Transformer (offline, Jax/Flax)
    └── pebble/                   # PEBBLE / B-Pref (online, PyTorch)
```

### `algorithms/preference_transformer/` — Preference Transformer

Transformer reward model that captures non-Markovian preferences, followed by offline RL (IQL)
on the relabelled dataset.

- Upstream: [csmile-1006/PreferenceTransformer](https://github.com/csmile-1006/PreferenceTransformer)
  (Kim et al., *Preference Transformer: Modeling Human Preferences using Transformers for RL*, ICLR 2023)
- Entry points: `JaxPref/new_preference_reward_main.py` (reward model), `train_offline.py` (IQL)
- Local additions: LunarLander support, synthetic / noisy / scripted label generation, human-vs-synthetic
  study pipelines (`scripts/`), grid-noise and count-axis analysis and plotting
- Vendored dependencies: `d4rl/`, `flaxmodels/`, `viskit/`
- Notes: `WORKLOG.md`, `HANDOFF_PT_SETUP.md`, `DIFFICULTY_AXES.md`, `WORKFLOW_grid_noise.md`

> ⚠️ `new_preference_reward_main.py` unpacks `sorted(os.listdir(<data_dir>/<env>))` into exactly three
> files. Every env folder of preference queries must contain **exactly** the three index/label files.

### `algorithms/pebble/` — PEBBLE (B-Pref)

Online preference-based RL: SAC with unsupervised pre-training, reward learning from a scripted
teacher or from live human labels, and relabelling of the replay buffer.

- Upstream: [rll-research/BPref](https://github.com/rll-research/BPref)
  (Lee et al., *B-Pref: Benchmarking Preference-Based Reinforcement Learning*, NeurIPS 2021 Datasets & Benchmarks)
- Entry points: `train_PEBBLE.py`, `train_SAC.py`, `train_PrefPPO.py`
- Local additions: LunarLander environment, web-based human labeling (`label_web.py`, `human_label.py`),
  checkpoint/resume, save-at-episode-boundary, deterministic post-hoc evaluation (`eval_saved_policy.py`),
  plotting (`plots/`), Slurm launchers (`run_cc_*.sh`, `scripts/lunar_lander/`)
- Vendored dependencies: `custom_dmcontrol/`, `custom_dmc2gym/`, `stable_baselines3/`, `rlkit/`

## Data and outputs

Nothing generated is tracked here. Run logs, checkpoints, reward models, preference-query sets,
human labels and figures live on cluster storage (`$SCRATCH` for active runs, project space for
anything kept), and `.gitignore` keeps them out of the repository. **Human-subject preference data
must never be committed.**

### Local data links

PT expects four entries in `algorithms/preference_transformer/` that point at cluster storage. They are
gitignored, so each fresh clone has to recreate them:

```bash
cd algorithms/preference_transformer
for d in human_label logs reward_model figures; do ln -s /scratch/$USER/learning-from-preferences/pt/$d $d; done
```

Run data lives in `/scratch/$USER/learning-from-preferences/`:
- `pt/`: offline datasets, IQL runs, reward models, logs and preference-query sets
- `pebble/`: PEBBLE run directories, previously `compare_runs/pebble`

The true-reward SAC baseline still writes to `$SCRATCH/compare_runs/sac_truereward/`.

`human_label/` holds preference-query sets: segment-pair indices plus a verdict, most of them synthetic.
Each env folder in it must contain exactly three files (see the warning above).

## Provenance

The repository was created on 2026-10-05 by copying the tracked files from two project-specific repos.
Their Git history was not carried over. Both source repos are tagged `pre-generalization-snapshot`:

| Folder | Source repo | Commit |
|---|---|---|
| `algorithms/preference_transformer/` | `M-rzyh/PT` (branch `offline-mixture-mr-and-grid-plots`) | `69d3f0d` |
| `algorithms/pebble/` | `M-rzyh/BPref3` (branch `main`) | `3d6cc67` |

## License

Each algorithm folder keeps its upstream `LICENSE` file. Please cite the original papers when you
use the corresponding code.
