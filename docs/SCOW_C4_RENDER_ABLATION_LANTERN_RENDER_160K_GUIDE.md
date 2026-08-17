# Lantern material-render 40k→160k SCOW guide

This incremental campaign resumes only `material_render` from the verified Job 37581
40k checkpoint. It does not modify the 20k or 40k outputs.

## Frozen schedule

- source: Job 37581 `material_render@40k`
- endpoint: 160k (120k continuation updates)
- observations: 60k, 80k, 100k, 120k, 140k, 160k
- full checkpoints: 80k, 120k, 160k
- loss, learning rates, rig, quantization, safety, seed stream: unchanged
- formal holdout: forbidden

## Install the incremental bundle

Upload the ZIP to:

```text
$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-render-160k-v1.zip
```

Then run in the existing campaign root that contains Job 37581:

```bash
ROOT="$HOME/projects/cg_frontier/campaigns/c4-render-ablation-20k-v1"
ZIP="$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-render-160k-v1.zip"
PATCH="$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-render-160k-v1"

cd "$ROOT"
sha256sum "$ZIP"
mkdir -p "$PATCH"
unzip -q "$ZIP" -d "$PATCH"
cd "$PATCH"
sha256sum -c PAYLOAD.MANIFEST.sha256
cd "$ROOT"
sha256sum -c "$PATCH/PATCH_BASELINE.sha256"
cp -a "$PATCH/payload/." "$ROOT/"
python3 scripts/verify_c4_render_ablation_lantern_render_160k_bundle.py \
  --bundle "$ZIP" \
  --installed-root "$ROOT"
```

## Submit

Every submit command refuses to run while another job is pending or running.

```bash
cd "$ROOT"
squeue -u "$USER"
bash scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh preflight
```

After the preflight job exits and its marker is verified:

```bash
PF=<PREFLIGHT_JOB_ID>
python3 -m json.tool \
  "outputs/remote/c4-render-ablation-lantern-render-160k-v1/${PF}/preflight_verified.json"
squeue -u "$USER"
bash scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh formal "$PF"
```

If a formal run fails after writing an 80k or 120k checkpoint, diagnose the error first.
Only then resume from the newest verified checkpoint:

```bash
bash scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh \
  resume <FAILED_JOB_ID> <80000_OR_120000> <PREFLIGHT_JOB_ID> yes
```

Do not blindly resubmit after NaN, OOM, hash mismatch, or environment drift.
