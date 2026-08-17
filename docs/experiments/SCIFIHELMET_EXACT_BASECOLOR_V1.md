# SciFiHelmet strict BaseColor experiment v1

## Scope

This experiment compares one unconstrained and two strictly constrained
RGBA8 encodings under the same 31-camera, 6-light train-only contract:

- `U0-unconstrained`: initialized byte-for-byte from `S`, then trains all four
  bytes and all seven affine decoder rows.
- `S-separated`: stores linear-Q8 RGB plus one residual byte.
- `M-mixed`: stores a bounded integer shear code along an affine BaseColor
  null direction.

The implementation is independent of the historical affine training stack.
It reuses only the stable glTF/Core-4 loaders, GBuffer construction, tangent
normal transform, and GGX renderer. The experiment was recovered from the
archived branch into the current repository as frozen reproduction code; it
does not replace the multi-asset C4 render-ablation mainline.

The formal holdout, SCOW, Unreal Engine, and pushing are outside this run.

## Reproduction

Use the repository virtual environment from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/run_scifihelmet_exact_basecolor.py audit
.\.venv\Scripts\python.exe scripts/run_scifihelmet_exact_basecolor.py preflight
.\.venv\Scripts\python.exe scripts/run_scifihelmet_exact_basecolor.py train
.\.venv\Scripts\python.exe scripts/run_scifihelmet_exact_basecolor.py report
```

`train` stages every candidate at 1k before any candidate advances to 5k.
The 1k/5k/10k checkpoints are immutable and a rerun must use a new output
root. All experiment artifacts are under the ignored directory
`outputs/scifihelmet_exact_basecolor_v1/`.

Runtime exports can be checked in a source-texture-free process with:

```powershell
.\.venv\Scripts\python.exe scripts/verify_scifihelmet_exact_basecolor_export.py `
  outputs/scifihelmet_exact_basecolor_v1/training/S-separated/export
```

The runtime verifier reads only the RGBA PNG, one `4→7` affine decoder NPZ,
and their manifest.

## Audit result

- Linear-Q8 BaseColor has 2,964 unique colors and exact affine rank 3.
- The unavoidable source-linear to Q8 floor is MAE `0.000990125`.
- The selected mixed code is `w=(-1,-1,-1,1)`, `t0=255`.
- Its minimum states per observed color are 160; frequency-weighted mean
  capacity is `7.92327` bits.
- The manifest retains all 3,088 feasible lattice candidates and the complete
  per-color `K(C)` distribution.

Preflight confirmed finite, nonzero auxiliary/render gradients, an exactly
zero BaseColor Jacobian for `S/M`, and bit-identical direct step 11 versus
`10→checkpoint→resume→11` states for all candidates.

## 10k result

| Metric | U0 | S | M |
|---|---:|---:|---:|
| BaseColor byte-exact texels | 27.07% | 100% | 100% |
| Linear-Q8 UV max error, 1M probes | 0.030385 | 0 | 2.37e-7 |
| Chroma retention | 1.18276 | 1.00000 | 1.00000 |
| Mean normal angle | 12.3315° | 12.7772° | 12.3958° |
| Roughness MAE | 0.05599 | 0.10503 | 0.10543 |
| Metallic MAE | 0.00939 | 0.01998 | 0.01932 |
| 31×6 mean HDR MAE | 0.011827 | 0.015561 | 0.015558 |
| 31×6 worst HDR MAE | 0.035812 | 0.044875 | 0.046507 |
| Mean display SSIM (global) | 0.851586 | 0.833160 | 0.833908 |

`M` improves paired mean HDR by only `0.0288%`; the camera/light bootstrap
95% interval is `[-0.433%, 0.491%]`, and its worst-pair HDR is 3.64% worse
than `S`. It therefore fails the frozen 1% improvement, confidence interval,
and worst-pair gates.

The required conclusion is:

> No mixed lattice advantage was observed; prefer `S-separated` for its
> larger uniform residual capacity and simpler decoder.

`U0` did not reproduce grayification: chroma retention increased rather than
falling below 0.5, and only 0.81% of chromatic texels lost at least 25% chroma.
Shared affine freedom alone is therefore insufficient to reproduce the old
failure; the old initialization or optimization geometry is also necessary.

## Diagnostics and failure record

The first CUDA preflight exposed NaN gradients at black display pixels. The
cause was the inactive power branch of a `torch.where` sRGB transform:
`pow(0, 1/2.4)` has an infinite derivative and contaminated the selected
linear branch through `0*inf`. Clamping the power-branch input to the sRGB
breakpoint preserves the forward value and makes its gradient finite. A
regression test covers black and near-black HDR inputs.

Primary evidence:

- `outputs/scifihelmet_exact_basecolor_v1/audit/lattice_manifest.json`
- `outputs/scifihelmet_exact_basecolor_v1/preflight/preflight_report.json`
- `outputs/scifihelmet_exact_basecolor_v1/training/training_report.json`
- `outputs/scifihelmet_exact_basecolor_v1/report/final_summary.json`
- `outputs/scifihelmet_exact_basecolor_v1/report/checkpoint_verification.json`
- `outputs/scifihelmet_exact_basecolor_v1/report/runtime_export_verification.json`
- `outputs/scifihelmet_exact_basecolor_v1/report/diagnostics_manifest.json`

The report directory also contains source-float/Q8-floor BaseColor images,
the U0/S/M BaseColor comparison, and training trajectories. Each candidate's
`diagnostics/` directory contains decoded BaseColor, normal, roughness,
metallic, and a four-panel material error atlas.

## UE preview export

The local UE preview bundle is generated with:

```powershell
.\.venv\Scripts\python.exe scripts/export_scifihelmet_exact_basecolor_ue.py
```

The isolated UE setup script is
`ue_demo/CGCompressionDemo/Content/Python/setup_scifihelmet_exact_basecolor_v1.py`.
It imports all three candidates below `/Game/CGCompression/ExactBaseColorV1`,
creates one-sample Custom materials, and duplicates `MaterialLab` into the
dedicated map `/Game/CGCompression/ExactBaseColorV1/Maps/ExactBaseColorV1`.
The source map is never saved by this setup.

The two clean headless setup passes completed successfully. UE texture
readback for U0/S/M is byte-identical to each source RGBA PNG (`max_abs=0`),
and the logs contain no `M_ExactBC_*` compile errors. Open the dedicated map
and inspect `Helmet_Reference`, `Helmet_ExactBC_U0`, `Helmet_ExactBC_S`, and
`Helmet_ExactBC_M`; `M_ExactBC_S` remains the recommended strict material.
