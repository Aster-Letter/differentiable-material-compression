# ADR 0003: Quality-first four-corner decode-then-material-filter rendering

- Status: Accepted; offline implementation complete, UE acceptance pending
- Date: 2026-08-04
- Scope: SciFiHelmet C4/C5 learned material compression

## Context

The legacy single-sample path evaluates a nonlinear decoder after bilinear filtering a latent texture. ARC can constrain some local ReLU crossings, but its complete run remains an ablation of that data flow: later optimization can reintroduce dark points and rectangular blocks even while the aggregate training loss falls.

The course objective is learned storage of BaseColor, tangent-space normal, roughness and metallic with real-time decoding before GBuffer material writes, judged primarily by final PBR appearance. A single RGBA8 resource, one bilinear sample and a 4→8→7 decoder are not course requirements. They remain useful cost and failure controls.

## Decision

1. Keep the legacy and `canonical_renderer_v2` single-sample paths read-only as comparison-only ablations.
2. Use an independent `decode_then_filter_renderer_v1` for the quality-first mainline.
3. Store a 2048×2048 C4 UNORM8 learned latent first. C5 is the representation upper bound and is only eligible after the C4 capacity route is evaluated.
4. Quantize stored texels before lookup. At LOD0 with Wrap addressing and no mipmaps, fetch the four bilinear footprint corners, run the shared decoder independently at each corner, and apply Core-4 postprocessing per corner.
5. Filter BaseColor in linear RGB, roughness in the existing perceptual-roughness scalar domain, and metallic linearly. Filter the four positive-hemisphere tangent normals as XYZ vectors, then normalize the filtered tangent normal exactly once before the shared TBN and GGX path.
6. Compare `C4-DTF-16-ReLU` and `C4-DTF-16-SiLU` in paired 10k prechecks from one immutable latent/affine initialization. Only a demonstrated quality advantage selects SiLU for the full candidate.
7. Train the selected `C4-DTF-16` for 80k steps: 0–15k material continuous-field pretraining, 15k–65k render-first joint training and 65k–80k low-learning-rate polish.
8. Use the fixed generic 45/35/10/10 mixture of screen-space render, uniform UV-chart subpixel, generic high-gradient/material-boundary and texel-center quantization-anchor samples. Do not inherit ARC, commutativity, fixed-region dark-tail or component-specific objectives.
9. Maintain `best-render` and `best-artifact-safe` checkpoint tracks. Training and selection data remain separate from the sealed final evaluation boundary.
10. If C4 does not reach the R0b visual-quality control, run a short `C4-DTF-32` capacity diagnostic. Material capacity evidence can justify widening; weak capacity benefit routes to fresh `C5-DTF-16`. C5 is the channel ceiling.

## Cost and storage accounting

`C4-DTF-16` has 471 float32 parameters (1,884 bytes), 432 decoder MAC per corner and 1,728 decoder MAC per shaded pixel before material filtering. It performs four point texel loads from one C4 resource and one tangent-normal normalization per pixel.

C4 and C5 theoretical UNORM8 raw sizes are 16 MiB and 20 MiB at 2048². Neither is reported as target-runtime actual resident memory until the engine/API format and bytes are measured. C5 may require two texture resources, so resource count and point loads are reported separately from logical channels.

## Selection and comparisons

- Traditional effective-channel packing is the storage/performance control.
- R0b is the filter-safe visual-quality control, not the recommended compressed deployment.
- Legacy and ARC are single-sample failure/ablation controls.
- Machine metrics prefilter candidates; anonymous UE comparison is reserved for an offline visual candidate and uses matched actor, transform, lighting, exposure and postprocess.

## Rejected alternatives

- Continuing to tune ARC-specific losses in the DTF mainline: rejected because DTF removes the nonlinear-after-filter seam structurally and should use a general objective.
- Preallocating C5's fifth channel to roughness or metallic: rejected because the fifth channel, if triggered, remains a learned latent.
- Increasing beyond C5 or dynamically adding task-specific ROI loss: rejected as unbounded scope expansion.
- Claiming theoretical channel bytes as actual engine residency: rejected because prior R0b evidence showed that logical and physical formats can differ.

## Consequences

The decoder executes four times per shaded pixel and quality must therefore be established before shader-cost optimization. In return, bilinear filtering happens in Core-4 material semantics instead of in a nonlinear latent chart, so the known center black-hole construction is eliminated by data flow rather than by a local penalty. Later DTF-8, pruning, distillation or repeated-computation reduction is conditional on C4 first reaching the visual target.

The NVIDIA repositories remain referenced only at the pinned revisions in `THIRD_PARTY_NOTICES.md`; no NVIDIA implementation is copied into this repository.

## Offline outcome

The paired 10k prechecks selected ReLU because SiLU did not show a clear overall quality advantage. Fresh `C4-DTF-16-ReLU` completed 80k steps, but remained below the documented R0b visual control. A fresh 10k `C4-DTF-32` diagnostic failed the significant-capacity-benefit rule, so the frozen route selected fresh `C5-DTF-16-ReLU` rather than a full width-32 run.

Fresh C5 completed 80k steps and selected step 80k for both checkpoint tracks. Its fixed evaluation reported HDR MAE `0.00237108`, display SSIM `0.99084407`, material composite `0.04546701`, novel-dark `0.00390848` and halo `0.00692059`, with no obvious rectangular black block. Relative to C4, C5 improved the primary render metrics and several material/artifact metrics, but used RGBA8+R8 resources and regressed normal mean, metallic MAE, novel-dark fraction and rectangular component area. It is therefore the offline quality candidate, not a claim of Pareto dominance or R0b equivalence. UE actual residency, anonymous visual acceptance and GPU timing remain required.
