# AGENTS.md — Reproduction & Engineering Guide for AI Agents

This file instructs autonomous coding and research agents on how to set up, verify, evaluate, and reproduce training for **Geo-ADMM-RL** faithfully without breaking physical fidelity, safety invariants, or evaluation hygiene.

---

## 1. Repository Architecture & Non-Negotiable Invariants

### Core Directory Map

| Path | Responsibility |
| --- | --- |
| `core/rstd.py` | Continuous-time analytical CPA distance and forbidden speed intervals (RSTD). |
| `core/conflict_graph.py` | 35 km conflict graph builder (existing + potential edges), graph coloring, connected components. |
| `core/native_vertical.py` | Exact B737 `0.05s` vertical acceleration & altitude-capture trajectory model (`4.1 μm` parity with native BlueSky). |
| `core/layer_search.py` | Batched paired and joint 3-aircraft discrete altitude layer search (`6 m` margin, `10 m` 3-ac reserve). |
| `core/admm_layer.py` | Distributed ADMM safety projection (Jacobi / colored schedules, warm-start vs cold-start, packet loss & delay). |
| `models/graphformer.py` | 3-layer Edge-Biased `GraphFormer` (`node=24`, `edge=18`, `width=256`, `heads=8`), sparse 4-mode SC-MoE `Actor` with exact left-right mirror equivariance (`--equivariant-actor`), and `TwinCritic` (`node=24+8=32`). |
| `envs/bluesky_plan_a.py` | Native BlueSky CPU batched rollout adapter (`BatchedBlueSkyEnv` from `bluesky-for-rl` `plan-a-batched-cpu`). |
| `envs/bluesky_plan_b.py` | Accelerator-resident vectorized rollout adapter (`TorchVecAirspaceEnv` from `bluesky-for-rl` `plan-b-pytorch-gpu`). |
| `envs/bluesky_safety.py` | Runtime wrapper combining actor output, ADMM projection, directional heading guard, layer search, and MPC gap control. |
| `envs/bluesky_traffic.py` | Traffic scenario generators (`crossing3`, `crossing4`, `crossing6`, `head_on`, `yaw_plus15`, `yaw_minus15`, `full_mixture`, held topologies). |
| `train/td3.py` | Multi-step Twin-Critic TD3 learner, stratified/terminal/topology replay, projected-action STE, mirror replay. |
| `train/bluesky_td3.py` | Unified RL training CLI (`--plan a|b|native`). |
| `train/bluesky_stage2.py` | Stage 2 gated admission, dual-panel native dev checkpoint selection, and early stopping. |
| `eval/bluesky_pair.py` | Official native `0.05s` frame-by-frame paired evaluator across the 6 scenario families. |
| `eval/admm_gap_mpc.py` | Local MPC controller executed during ADMM communication gaps (e.g., 12 s / 18 s / 30 s cadence). |
| `eval/stage_acceptance.py` | Frozen operational acceptance gates (`EXACT_NATIVE_PROTOCOL`, `MPC_COMMUNICATION`). |
| `eval/bluesky_figure4_matrix.py` | Exp 1 Figure 4 benchmark (`geo` vs `sarl-kss` vs `hold-kss`, 15 manifests). |
| `eval/bluesky_paper_*.py` | Paper evaluation suite: Exp 2 & 3 (`generalization`), Exp 4 (`exp4`), Exp 5 (`exp5`), Exp 6 (`exp6`), and SHA-256 auditor (`audit`). |
| `configs/b737_legacy_cruise.json` | Measured native BlueSky B737 legacy cruise calibration (`a=2.0 m/s^2`, `bank=25 deg`, `vs=7.62 m/s`). |
| `tests/test_geometry.py` | 80 unit, geometry, safety-layer, equivariance, and protocol regression tests (including delayed single-send transport, zero-communication decentralized MPC, inertia-aware `altitude_admm`, and cross-track zero-init weight migration). |

### Physical & Protocol Invariants (DO NOT VIOLATE)

- **Separation thresholds**: Horizontal `< 5.0 km` AND Vertical `< 300 ft (91.44 m)` = collision.
- **Arrival / Exit thresholds**: `< 5.0 km` to destination = arrival; `> 20.0 km` cross-track deviation = exit.
- **Official evaluation step size**: Native BlueSky `bs.sim.step()` with `fast=False`, `sim_dt = 0.05 s`, frame-by-frame event check (`0.05 s`), `control_dt = 6.0 s`, `max_steps = 400`.
- **Action cost proxy**: Per-aircraft cumulative normalized command magnitudes weighted by `(speed, altitude, heading) = (1, 2, 4)`.
- **Mirror equivariance**: Under `--equivariant-actor`, left-right mirror reflection must produce `0` error on raw actor outputs, discrete mode routing, and projected commands; ADMM warm-start heads use `even` reflection parity.
- **Seed separation**:
  - **Development panels** (checkpoint selection & tuning): `61231–61235` and `75121–75125`.
  - **Frozen formal holdout** (final reporting ONLY — never tune on these): `12001–12005`.

---

## 2. Environment & Dependency Setup

### Python Requirements

- Python `3.10` – `3.12`
- `torch` (CPU, CUDA, or PPU PyTorch build), `numpy`, `scipy`, `pytest`, plus BlueSky's lightweight runtime dependencies (`msgpack`, `pyzmq`, etc., listed in `bluesky-for-rl`).

### Clone External BlueSky Accelerator Branches

This repo intentionally excludes `third_party/`. Clone the two branches of [`Reek12138/bluesky-for-rl`](https://github.com/Reek12138/bluesky-for-rl) into `third_party/`:

```bash
mkdir -p third_party
git clone -b plan-a-batched-cpu https://github.com/Reek12138/bluesky-for-rl.git third_party/bluesky-plan-a
git clone -b plan-b-pytorch-gpu https://github.com/Reek12138/bluesky-for-rl.git third_party/bluesky-plan-b
```

Verified reference commits:
- `plan-a-batched-cpu`: `406a22fbf31c4e141442e1fc745b6b3c1c35fbd2`
- `plan-b-pytorch-gpu`: `f38a7d9b200750b226c49af149b2d0b49b0e4772`

### `PYTHONPATH` Selection Rule

Because both branches expose top-level package `bluesky`, set `PYTHONPATH` strictly per backend:
- **Native evaluation / Plan A training (`--plan a` or `--plan native`)**:
  ```bash
  export PYTHONPATH="third_party/bluesky-plan-a:."
  ```
- **Plan B GPU/PPU training (`--plan b`) and full 80-test pytest run**:
  ```bash
  export PYTHONPATH="third_party/bluesky-plan-b:."
  ```

### Hardware Device Note (`--device cpu | ppu | cuda`)

- All training and evaluation entrypoints accept `--device cpu`, `--device ppu`, or `--device cuda`.
- `--device ppu` enforces the strict single **PPU-ZW810E** allocation guard used for the published benchmarks; `--device cuda` runs directly on any NVIDIA CUDA GPU (`cuda:0`) without the PPU device-name check.

### Navigation Cache Deduplication Tip

BlueSky generates a ~15.73 MiB navdata cache per fresh working directory. Reuse a single pre-generated navdata cache file across evaluation runs by passing `--navdata /path/to/navdata.dat` to avoid creating gigabytes of duplicate caches.

---

## 3. Fast Verification Gate (Run Before & After Any Code Edit)

### 3.1 Full Unit & Regression Suite (80 Tests)

```bash
PYTHONPATH=third_party/bluesky-plan-b:. python -B -m pytest tests -q -p no:cacheprovider
```

Expected: `80 passed` (always pass `tests` explicitly so pytest does not collect duplicate `third_party/` BlueSky test suites; if `bluesky.rl_gpu` is absent from `PYTHONPATH`, `79 passed, 1 skipped`).

### 3.2 Geometry Kernel Micro-Benchmark

```bash
python -B -m eval.benchmark_geometry --device cpu --batch 32 --repeat 5
# On accelerator (PPU):
python -B -m eval.benchmark_geometry --device ppu --batch 512 --repeat 20
```

### 3.3 Simulator Parity Checks

```bash
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.verify_plan_a
PYTHONPATH=third_party/bluesky-plan-b:. python -B -m eval.verify_plan_b_ppu
```

---

## 4. Reproducing Inference & Evaluation

### 4.1 Official 6-Scenario Paired Native `0.05s` Evaluation (`eval.bluesky_pair`)

This reproduces the exact **98.61% (`355/360`)** release protocol (`EXACT_NATIVE_PROTOCOL | MPC_COMMUNICATION` from `eval/stage_acceptance.py`):

```bash
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_pair \
  --baseline <baseline.pt> \
  --candidates <release_15k.pt> \
  --output outputs/eval_formal_holdout \
  --device cpu \
  --seed 12001 --seeds 5 \
  --aircraft 12 --control-dt 6.0 --max-steps 400 \
  --admm-k 2 --admm-fallback 0 --dense-extra-rounds 2 \
  --directional-heading \
  --batched-layer-search --early-layer-search \
  --exact-layer-forecast --batched-layer-margin 6.0 \
  --joint-layer-search --joint-layer-reserve 10.0 \
  --admm-communication-period 12.0 --admm-gap-policy mpc \
  --ablation --ablate-baseline \
  --parallel-pairs 1
```

- For **development validation** during checkpoint selection, swap `--seed 12001` to `--seed 61231` (Panel 1) and `--seed 75121` (Panel 2).
- Add `--navdata <path>` to reuse a cached BlueSky navdata file.
- Add `--graph-statistics` or `--track-yaw-recovery` for diagnostics (note `--track-yaw-recovery` only applies to `--family yaw_plus15 yaw_minus15`).

### 4.2 Verifying the Stage Acceptance Gate

```bash
python -B -m eval.stage_acceptance <outputs/eval_formal_holdout/comparison.json>
```

The release gate checks:
- Exact protocol match (`sim_dt=0.05`, `control_dt=6.0`, `detection_metres=35000`, `equivariant_actor=True`, `admm_warmstart_reflection="even"`, `batched_layer_margin_m=6.0`, `joint_layer_reserve_m=10.0`, `admm_communication_period=12.0`, `admm_gap_policy="mpc"`).
- All 6 families × 5 seeds (`360` aircraft) accounted for with zero missing outcomes.
- Operational pass threshold (`> 95%` arrival, `<= 3%` collision).

### 4.3 Paper Benchmark Suite (Exp 1 – Exp 6 & Integrity Audit)

All paper matrix runners require a verified `acceptance.json` produced by `eval.stage_acceptance` and the matching release checkpoint:

```bash
# Exp 1: Figure 4 three-arm native comparison (Geo vs SARL+KSS vs Hold+KSS, 15 manifests)
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_figure4_matrix \
  --acceptance <acceptance.json> \
  --checkpoint <release_15k.pt> \
  --manifests <figure4_manifests_dir> \
  --history <sarl_history_dir> \
  --sarl-checkout <sarl_repo_dir> \
  --navdata <navdata_cache> \
  --output outputs/exp1 \
  --device cuda:0

# Exp 2 & 3: Zero-shot topology generalization (13/15/17 routes) & scaling (30/50/80 ac)
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_paper_generalization \
  --acceptance <acceptance.json> \
  --checkpoint <release_15k.pt> \
  --navdata <navdata_cache> \
  --output outputs/exp2_3 \
  --device ppu

# Exp 4: 30-ac head-on stress, +/-15 deg yaw, and 25 environmental perturbations
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_paper_exp4 \
  --acceptance <acceptance.json> \
  --checkpoint <release_15k.pt> \
  --navdata <navdata_cache> \
  --output outputs/exp4 \
  --device ppu

# Exp 5: 61-config ADMM communication matrix (rounds, Hold vs MPC cadence, loss x delay, coloring)
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_paper_exp5 \
  --acceptance <acceptance.json> \
  --checkpoint <release_15k.pt> \
  --navdata <navdata_cache> \
  --output outputs/exp5 \
  --device ppu

# Exp 6: Ablations (24-step vs 1-step return, no projection, no potential edges, KSS substitution)
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m eval.bluesky_paper_exp6 \
  --acceptance <acceptance.json> \
  --checkpoint <release_15k.pt> \
  --manifests <figure4_manifests_dir> \
  --history <sarl_history_dir> \
  --sarl-checkout <sarl_repo_dir> \
  --navdata <navdata_cache> \
  --one-step-checkpoint <one_step_control.pt> \
  --control-config <multi_step_config.json> \
  --one-step-config <one_step_config.json> \
  --output outputs/exp6 \
  --device ppu

# Full SHA-256 Artifact & Protocol Audit across Exp 1–6
python -B -m eval.bluesky_paper_audit outputs/paper_suite_dir
```

---

## 5. Reproducing Training (Stage 0 → Stage 1 → Stage 2)

### Stage 0: Analytical Expert Data Generation & Behavior Cloning (BC Initialization)

1. Generate verified continuous-time expert labels (speed + altitude tiers):
   ```bash
   python -B -m train.pilot_data \
     --reference <reference_geometry_solver_dir> \
     --output outputs/stage0/data \
     --train 2048 --validation 384 --ood 384 \
     --aircraft 6 --workers 6 --with-altitude
   ```
2. Train the initial `GraphFormer` Actor via Behavior Cloning:
   ```bash
   python -B -m train.behavior_clone \
     --data outputs/stage0/data \
     --output outputs/stage0/bc_model \
     --device ppu --epochs 12 --batch 32 --width 256 --seed 21701
   ```
   Produces `outputs/stage0/bc_model/actor.pt` (used to bootstrap RL).

### Stage 1: Plan B GPU/PPU Equivariant Multi-Step TD3 (Reproduces the 98.61% Policy)

Run 512 device-resident airspaces (`dt=0.25s`, calibrated B737 legacy cruise, 35 km detection graph, exact left-right mirror equivariant actor, 24-step/144 s multi-step return, projected target + STE, exact 0.05s vertical layer forecast with 6 m margin & 10 m 3-ac reserve):

```bash
PYTHONPATH=third_party/bluesky-plan-b:. python -B -m train.bluesky_td3 \
  --plan b --device ppu \
  --output outputs/stage1_equivariant_20k \
  --initialize <init_checkpoint.pt> \
  --initialize-critic \
  --equivariant-actor \
  --family full_mixture \
  --timed-departures --physics64 \
  --envs 512 --aircraft 20 \
  --sim-dt 0.25 --control-dt 6.0 --max-steps 400 \
  --performance-calibration configs/b737_legacy_cruise.json \
  --detection-km 35 \
  --admm-k 2 --admm-fallback 0 --dense-extra-rounds 2 \
  --directional-heading \
  --batched-layer-search --early-layer-search \
  --exact-layer-forecast --batched-layer-margin 6.0 \
  --joint-layer-search --joint-layer-reserve 10.0 \
  --projected-target-action --projected-actor-ste \
  --executed-action-critic --mirror-replay \
  --stratified-replay --return-window 24 --multi-step-return \
  --crossing-exploration-steps 8 \
  --shaped-reward --lookahead-weight 0.7 --raw-risk-weight 0.75 \
  --navigation-weight 1.5 --projection-weight 0.12 \
  --coordination-weight 0.15 \
  --batch 64 --inference-batch 256 --updates 10 --replay 8192 \
  --learning-rate 1.2e-4 --cosine-updates 20000 \
  --target-updates 20000 --checkpoint-every 5000
```

*(Set `--coordination-weight 0.0` to reproduce the historical frozen-warmstart `98.61%` checkpoint; keep `--coordination-weight 0.15` and optionally add `--admm-communication-period 12 --admm-gap-policy mpc --selected-altitude-observation --cross-track-observation` for joint ADMM warm-start + decentralized MPC co-training.)*

**Checkpoint Selection Protocol (Critical)**:
- Evaluate every `5,000` updates (`5k`, `10k`, `15k`, `20k`) on **both** native `0.05s` development panels (`61231–61235` and `75121–75125`) with `12 s MPC` AND run the 4-case native Critic Q ranking audit (`eval.bluesky_team_preference_audit`).
- In the verified run (`b35-equivariant-even-admm-20k`), the **`15,000`-update checkpoint** achieved the peak: `4/4` native Critic Q, `355/360` on Panel 1, `350/360` on Panel 2, and **`355/360` (98.61%, 4 col)** on formal holdout `12001–12005`, whereas `20k` slightly regressed on yaw recovery. Always select checkpoints by native dev panels, never by final step count.

### Stage 2: Plan A Native `0.05s` High-Fidelity Refinement (`train.bluesky_stage2`)

Optional short high-fidelity refinement using native Plan A (`9 workers × 8 envs` configured internally, `dt=0.05s`, LR = `1/5` of Stage 1 = `2.4e-5`), gated by dual-panel native development evaluation and 4-case Critic Q early stopping (`< 3/4` or 2 consecutive regressions retains the incoming Stage 1 checkpoint):

```bash
PYTHONPATH=third_party/bluesky-plan-a:. python -B -m train.bluesky_stage2 \
  --gate <stage2_native_admission.json> \
  --checkpoint <stage1_15k.pt> \
  --frozen-baseline <frozen_bc_baseline.pt> \
  --raw-reference <prior_release_checkpoint.pt> \
  --output outputs/stage2_refine \
  --device ppu \
  --short-refine --equivariant-refinement \
  --episodes 600 --every-episodes 200 \
  --detection-km 35 --return-window 24 --multi-step-return \
  --initialize-critic --batched-layer-search --executed-action-critic \
  --correlated-yaw --topology-replay \
  --navdata <navdata_cache>
```

*(Inspect `python -B -m train.bluesky_stage2 --help` for optional `--q-audit`, `--q-source`, `--q-crossing-four`, and `--q-crossing-six` frozen Q-ranking hooks.)*

---

## 6. Hard-Won Engineering Pitfalls for Agents

1. **NEVER report Plan B (`dt=0.25s`) or 6 s control-frame checks as final metrics.** Plan B can hide brief mid-step intrusions; all publishable metrics MUST come from native `bs.sim.step()` (`dt=0.05s`, frame-by-frame audit).
2. **NEVER disable `--exact-layer-forecast` or `--batched-layer-margin 6.0` in release evaluations.** Linear altitude extrapolation underestimates B737 climb inertia by `2 mm – 25 cm`, causing false-safe grazing collisions during dense crossings.
3. **NEVER break `--equivariant-actor` or ADMM warm-start reflection parity.** After any change to `envs/observations.py`, `envs/bluesky_graph.py`, or `models/graphformer.py`, verify exact `0` mirror error via `pytest` and a frozen-state symmetry audit.
4. **Watch for yaw recovery drift on long runs.** Prolonged training without monitoring `yaw_plus15` / `yaw_minus15` on native dev panels can cause delayed heading commands past the `0.05` STE deadband and trigger 20 km cross-track exits (as observed at update `10k` and Stage 2 `200/400` ep). Keep dev panel gating active.
5. **Preserve 1-step control parity for Exp 6.** When running the Exp 6 1-step return ablation (`train/bluesky_one_step_control.py`), keep identical initialization, seed, replay maturity (`24` decisions), decision budget, and update count — only change the Critic return horizon.
6. **Keep git clean.** Never commit `third_party/`, `*.pt` weights, `navdata` caches, logs, or tokens/credentials. Always run `python -B` and `-p no:cacheprovider` so temporary caches do not pollute the workspace.
