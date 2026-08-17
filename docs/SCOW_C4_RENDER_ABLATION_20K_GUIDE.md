# SCOW C4 render-ablation 20k manual run guide

This campaign is intentionally operated by the user through the official SCOW
Shell. It does not require, request, or permit browser automation.

## Frozen experiment

- Assets, in submission order: `Corset`, `Lantern`, `BoomBox`.
- Arms per asset: `material_only`, then `material_render`.
- Both arms start fresh from the same global raw-q4 PCA/RGBA8/4-to-7 affine
  parent and consume the same explicit Torch RNG sequence for texels, cameras,
  and lights.
- Formal endpoint: 20,000 steps. Observations: 1k/5k/10k/15k/20k. Full
  optimizer/RNG checkpoints: 10k/20k.
- Lantern excludes emissive from reference, PCA, and both arms. Its measured
  `max(RGB)>0.05` emissive fraction is `0.03266143798828125`; conclusions only
  cover BaseColor, tangent Normal, Roughness, and Metallic.
- BoomBox likewise excludes its previously reviewed sparse emissive accent
  (`0.004467010498046875`).
- Formal holdout and Unreal Engine are forbidden for this campaign.

## Upload and verify

Upload `c4-render-ablation-20k-v1.zip` to:

```text
$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-20k-v1.zip
```

In the official SCOW Shell:

```bash
cd "$HOME/projects/cg_frontier"
mkdir -p campaigns/c4-render-ablation-20k-v1
unzip -q transfers/incoming/c4-render-ablation-20k-v1.zip \
  -d campaigns/c4-render-ablation-20k-v1
cd campaigns/c4-render-ablation-20k-v1
sha256sum -c payload.MANIFEST.sha256
python3 scripts/verify_c4_render_ablation_bundle.py
```

All commands below must be run from that campaign directory. The scripts reuse
the already validated environment at
`$HOME/projects/cg_frontier/.envs/cg-frontier-system-py312` unless `ENV_ROOT`
is explicitly set.

## Preflight

First confirm the queue is empty, then submit exactly one preflight:

```bash
squeue -u "$USER"
bash scripts/scow_submit_c4_render_ablation_20k.sh preflight
```

Record the returned job ID. Check it with:

```bash
squeue -u "$USER"
sacct -j <PREFLIGHT_JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
tail -n 100 logs/slurm/c4-render-ablation-20k-v1/c4-ra20-preflight.<PREFLIGHT_JOB_ID>.out
tail -n 100 logs/slurm/c4-render-ablation-20k-v1/c4-ra20-preflight.<PREFLIGHT_JOB_ID>.err
```

Proceed only if `sacct` reports `COMPLETED`, stderr is empty, and this marker
exists:

```text
outputs/remote/c4-render-ablation-20k-v1/<PREFLIGHT_JOB_ID>/preflight_verified.json
```

## Formal serial submissions

Only submit the next asset after the previous job reaches a terminal state and
`squeue -u "$USER"` is empty:

```bash
bash scripts/scow_submit_c4_render_ablation_20k.sh formal Corset  <PREFLIGHT_JOB_ID>
bash scripts/scow_submit_c4_render_ablation_20k.sh formal Lantern <PREFLIGHT_JOB_ID>
bash scripts/scow_submit_c4_render_ablation_20k.sh formal BoomBox <PREFLIGHT_JOB_ID>
```

Each job requests `Students`, one RTX5090, four CPUs, and four hours. It does
not request memory, account, or QOS. A successful job writes:

```text
outputs/remote/c4-render-ablation-20k-v1/<JOB_ID>/<ASSET>/
outputs/remote/c4-render-ablation-20k-v1/<JOB_ID>/<ASSET>-summary/
outputs/remote/c4-render-ablation-20k-v1/<JOB_ID>/formal_run_verified.json
logs/slurm/c4-render-ablation-20k-v1/c4-render-ablation-20k.<JOB_ID>.out
logs/slurm/c4-render-ablation-20k-v1/c4-render-ablation-20k.<JOB_ID>.err
```

Download the asset directory, summary directory, verifier marker, and paired
stdout/stderr before considering any remote cleanup. This package does not
authorize cleanup.

## Evidence-based 10k resume

Do not resubmit an unchanged failed job. Resume is allowed only when the source
job is `FAILED`, `TIMEOUT`, or `OUT_OF_MEMORY` and the affected arm contains a
valid `checkpoints/step_10000/checkpoint.pt` plus its progress snapshot:

```bash
bash scripts/scow_submit_c4_render_ablation_20k.sh resume \
  <ASSET> <material_only|material_render> <SOURCE_JOB_ID> <PREFLIGHT_JOB_ID>
```

The runner validates asset/config/PCA parent/rig/sampling lineage, restores both
optimizers and RNG, isolates stale products after 10k, completes the arm, and
then regenerates paired evidence. If the other arm had not started, it is run
fresh in the same resume allocation; an existing but incomplete other-arm root
fails closed and requires inspection. If no verified 10k checkpoint exists,
stop and inspect the failure instead of pretending the run is resumable.

## Interpretation

The summary reports `raw_q4 -> material_only`, `raw_q4 -> material_render`, and
`material_only -> material_render`. Only the last paired difference isolates
the contribution of render supervision. This is a deterministic single-seed
case study; do not claim variance estimates or statistical significance, and
do not force a winner when render and material metrics form a Pareto trade-off.
