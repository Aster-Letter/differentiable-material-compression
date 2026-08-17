# ADR 0004: C4 single-affine deployment mainline

- Status: Accepted; full-cube safety clause amended 2026-08-10
- Date: 2026-08-06
- Scope: Core-4 material compression deployment, training, PCA baseline, and UE integration
- Supersedes: ADR 0003 as the recommended deployment path; ADR 0003 remains a quality/cost control

## Context

The mentor clarified that the target is an extreme real-time game deployment: the stored representation is limited to four channels and the per-pixel decoder must be one linear mapping with no hidden layer. The existing C4/C5 decode-then-filter networks are useful quality evidence, but their four decoder evaluations and hidden layers no longer satisfy the intended compute budget.

The required comparison is not learned compression against the uncompressed material alone. Learned Linear must be compared with PCA under the same C4 storage and the same one-affine runtime path. This prevents decoder architecture, texture sampling, and postprocessing differences from confounding the representation comparison.

The prior yellow-tube defects also showed that aggregate texture or render losses do not by themselves guarantee spatially stable atlases. The mentor proposed TV regularization and an auxiliary cube render. These are treated as separately attributable extensions rather than silently added to the learned baseline.

## Decision

1. Store one per-asset 2048×2048 RGBA8 latent texture. C4 is a hard deployment ceiling.
2. Decode one hardware-filtered latent sample with a per-asset affine map `y = Wz + b`, where `W` is 7×4 and `b` is length 7. The 35 FP32 constants are supplied to one shared UE material implementation.
3. Interpret the seven affine outputs directly as linear BaseColor RGB, tangent-space normal XY, linear roughness, and linear metallic. Runtime sigmoid, tanh, conditional XY projection, and final normal normalization are not used.
4. Reconstruct positive normal Z with `sqrt(max(epsilon, 1-x²-y²))`; apply the UE tangent-normal Y bridge exactly once at the engine boundary.
5. Train with a smooth safe-by-construction parameterization over the full latent hypercube. Scalar coefficient budgets keep BaseColor, roughness, and metallic in `[0,1]`; a two-dimensional group budget keeps normal XY inside the unit disk. Export analytically folds the training representation into ordinary `W,b`. Certificates verify and fail closed; they never conditionally repair weights.
6. Use uniform-valid-texel 7D→4D PCA as P0. P0, L0, L1, and L2 share the exact deployment contract. Render-weighted PCA is an optional stronger baseline, not a replacement for P0.
7. Train three paired learned candidates from the same P0 state: L0 is the base material+helmet objective, L1 adds chart-aware latent TV, and L2 adds a canonical cube render. L3 combines TV and cube only if isolated evidence justifies it.
8. For L1, compute horizontal and vertical Charbonnier TV on fake-quantized/dequantized latent values, accepting only edges whose endpoints are valid and in the same UV chart. Normalize by accepted edge count. Calibrate the fixed coefficient once at P0 from gradient scale.
9. For L2, map the full `[0,1]²` texture domain to each of six canonical cube faces with explicit, consistent face orientation and tangent frame. Render six face-normal orthographic views under frozen generic light sampling, mask invalid atlas texels in screen space, normalize by valid pixels, and use an independent RNG stream. Calibrate the fixed coefficient once at P0 from gradient scale.
10. Export only the base RGBA8 image with `sRGB=false`. UE generates a non-sharpened linear-average mip chain, uses ordinary derivative LOD/filtering, and allows Texture Streaming. Mip-aware training is conditional on deployment evidence, because affine decoding commutes with linear filtering.
11. Preserve the C4/C5 DTF, R0b, ARC, legacy, baseline/reference, imported assets, and UE preview materials as read-only controls. Do not overwrite their outputs or reinterpret them as compliant deployment candidates.

## 2026-08-10 course-objective amendment

The course quality objective is the final rendered appearance of the model. For future lineages, frozen render HDR/SSIM and visual evidence are the primary optimization and selection evidence. Per-channel material and color metrics remain diagnostics and guards against conspicuous artifacts; they are not co-equal primary objectives.

Decision item 5 described a safe-by-construction experiment intended to simplify training and replace part of runtime normalization/postprocessing. Cross-asset evidence showed that full-cube safety increases seven-channel MAE to `4.8–50.2×` the corresponding raw q4 result on three simple nonmetal assets. Full-cube certification is therefore no longer required for new course experiments. Existing certified artifacts remain immutable safety/cost controls and valid historical evidence.

The hard deployment constraints remain one RGBA8 texture, one filtered sample, and one `4→7` affine decoder. A future raw-affine candidate must instead document finite behavior on valid latent texels and the hardware-filter reachable envelope, and must explicitly account for any bounded runtime output guard. This amendment does not retroactively change the contracts or interpretation of completed P0/L0/L1/L2 experiments.

## Experiment structure

- SciFiHelmet runs P0 and the complete L0/L1/L2 ablation. L3 is conditional.
- Three additional Core-4 assets run P0, L0, and the SciFiHelmet-selected extension. The set must contain a metal-dominant asset, a nonmetal/rough asset, and a high-frequency or complex-boundary asset.
- L0/L1/L2 first run to a common 40k endpoint. Continuations, if justified by the joint curves, are synchronized in 40k increments to 80k and 120k. A shared 160k continuation requires a new decision after reviewing 120k.
- Comparisons report continuous paired results and uncertainty. Planning expectations are not automatic winner gates.

## Runtime and storage accounting

The affine decoder has 35 FP32 constants (140 bytes) and 28 multiply-accumulates per shaded pixel. It uses one RGBA8 texture resource and one filtered sample. A 2048² base level is 16 MiB uncompressed; a complete uncompressed mip chain is approximately 21.33 MiB. These theoretical figures are reported separately from cooked bytes, actual residency, streaming behavior, shader compiler statistics, and measured GPU time.

UE 5.8 material compiler instruction/sample statistics are the provisional static evidence. They are not called native GPU ISA counts. The final meaning of “GPU instruction count” remains subject to mentor confirmation.

## Consequences

The deployment path becomes dramatically cheaper than the quality-first DTF path, while training may still be dominated by latent optimization, material rendering, loss evaluation, and the L2 auxiliary renderer. Quality is expected to decline relative to C5-DTF and must be judged primarily against same-cost PCA.

The global safe parameterization is deliberately conservative. Any later reachable-set relaxation, edge-aware TV, render-weighted PCA, alternate initialization, block-compressed latent, or mip-aware objective creates a separately named lineage and cannot mutate the main experiment in flight.

Formal holdout data remain sealed and are not used for fitting, candidate selection, protocol tuning, or this ADR.
