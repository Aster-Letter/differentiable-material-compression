# ADR 0002: Deployment-parity training with activation-region-coherent latent cells

- Status: Accepted for experiment
- Date: 2026-08-04
- Scope: SciFiHelmet single-sample RGBA8 material compression

## Context

The legacy prototype stores one 2048×2048 RGBA8 latent and evaluates a 4→8→7 ReLU decoder after one bilinear texture sample. Texel-center reconstruction can be correct while a bilinear latent mixture crosses a decoder activation boundary. The nonlinear composition therefore does not generally commute with filtering:

`decode(bilinear(latent)) != bilinear(decode(latent corners))`.

The observed yellow-tube dark dropouts and gray-panel dark cores/bright outlines are introduced at this filtering/decoder seam. Shared Core-4 postprocessing and GGX shading can increase their visual contrast, but are not their source. The existing nvdiffrast-based geometry/PBR path remains the physical reference and is not rewritten.

## Decision

1. Freeze the legacy renderer as a read-only reproduction path.
2. Make `canonical_renderer_v2` the only renderer for new training after it passes immutable forward, gradient and byte-determinism gates against the legacy path.
3. Preserve the deployment order: RGBA8 quantization at stored texels, one bilinear latent sample, one 4→8→7 ReLU decoder, one Core-4 postprocess, shared GGX PBR and display transform.
4. Train `DP-ReLU-Fresh` and `ARC-ReLU-Fresh` independently from the same seed-20260804 random latent/decoder state. Legacy, R0b and rejected checkpoints are comparison-only inputs.
5. Add a fixed dark-envelope loss to both candidates. Add a 2/255 normalized preactivation margin to ARC so the four quantized corners of a 2×2 latent cell remain on one side of every hidden-unit ReLU boundary.
6. Keep ARC-12 as a conditional capacity diagnostic; its 4→12→7 cost makes it ineligible for deployment selection.
7. Evaluate rectangular dark components before general material/render quality. A candidate with any area-8-or-larger rectangular dark component in either fixed artifact ROI fails regardless of aggregate image quality.

“Deployment-parity” here means parity of quantization, sampling, decoding and postprocessing order. It does not claim pixel identity between the Python GGX renderer and Unreal Engine Default Lit.

## Renderer takeover gates

- exact coverage/mask;
- UV maximum absolute error ≤ 2e-6;
- normal P99 angular error ≤ 0.01 degrees;
- Core-4 P99 absolute error ≤ 1e-5;
- linear HDR MAE ≤ 1e-4 and P99 ≤ 1e-3;
- display SSIM ≥ 0.9999;
- 32 fixed gradient probes finite, cosine ≥ 0.999 and relative L2 ≤ 1%;
- two reports byte-identical.

No threshold may be relaxed to permit takeover.

## Rejected alternatives

- Rewriting the complete PBR pipeline: rejected because it does not address the filtering/nonlinearity seam and would invalidate the shared reference.
- Initializing from legacy/R0b/filter-aware checkpoints: rejected because their learned latent topology may contain the failure being tested.
- Repeating Softplus or sigmoid candidates: rejected because prior valid runs showed that a smooth activation alone did not preserve material/render quality or remove the topology conflict.
- Adding channels, textures, mipmaps, BC formats, AO or a wider deployable network: outside this experiment's exact budget.
- Stochastic filtering: deferred because it introduces temporal noise and TAA dependence.

## Consequences

The training loop is more expensive because it samples deployed quantized latent neighborhoods and evaluates screen-space PBR supervision. In return, training and evaluation share one explicit deployment contract, ReLU crossings are measurable at the responsible 2×2 cells, and black-block removal is tested directly instead of inferred from global reconstruction metrics.

The NVIDIA repositories are referenced at the pinned revisions recorded in `THIRD_PARTY_NOTICES.md`; no NVIDIA implementation is copied into this repository.
