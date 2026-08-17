# Research plan

## Objective

Compress SciFiHelmet Core-4 material parameters into one 2048² RGBA8 latent texture and a 4→8→7 real-time decoder while preserving one bilinear texture sample, 16 MiB actual resident memory, 103 parameters, 412 bytes of weights and 88 MACs.

The primary acceptance criterion is interpolation safety: decoded subpixel samples must not introduce rectangular dark components, novel dark pixels or bright/dark material-boundary halos that are absent from the directly filtered reference material.

## Current hypothesis

The runtime evaluates a nonlinear function after latent-space filtering:

```text
RGBA8 texels → bilinear latent → tiny decoder → material postprocess → PBR
```

Texel-center reconstruction alone does not constrain the interior of a 2×2 latent cell. A ReLU preactivation can cross zero inside that cell and remove a BaseColor contribution even when all trained texel centers are acceptable.

## Current method

1. Build a canonical differentiable renderer whose material path is operation-for-operation compatible with the deployment contract.
2. Learn a new latent topology from a shared deterministic random initialization; do not warm-start from rejected repair candidates.
3. Compare deployment-parity ReLU training with an activation-region-coherent variant that constrains every hidden unit over each sampled 2×2 cell.
4. Reject candidates before Unreal Engine evaluation unless they pass deterministic artifact, material, rendering and structural-cost gates.

The formal final-evaluation partition remains sealed during architecture selection and is not distributed in this repository.

## References

- NVIDIA nvdiffrast supplies differentiable rasterization, interpolation and texture sampling primitives.
- NVIDIA nvdiffmodeling is used only as a read-only organizational and PBR reference.
- The local renderer remains an independent implementation governed by `docs/ASSET_CONVENTIONS.md`.
