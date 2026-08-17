# Third-party notices

本仓库的 MIT License 只覆盖本项目原创源码。以下依赖与资产继续适用各自许可证。

## NVIDIA nvdiffrast

- Repository: https://github.com/NVlabs/nvdiffrast
- Pinned revision: `253ac4fcea7de5f396371124af597e6cc957bfae`
- Role: differentiable rasterization, interpolation and texture sampling primitives
- Distribution: not vendored; installed from the upstream repository

## NVIDIA nvdiffmodeling

- Repository: https://github.com/NVlabs/nvdiffmodeling
- Reference revision: `9b2ba2eff83c7d90127f78c20773b06ddc3ae1db`
- Role: read-only reference for differentiable material/PBR organization
- Distribution: no NVIDIA source is copied into this repository; the local implementation is independent and follows the project's Core-4 conventions

## Khronos SciFiHelmet

- Repository: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SciFiHelmet
- License: CC0
- Distribution: asset files are not committed; the bootstrap script obtains them from the upstream repository

## Khronos glTF sample assets used in experiments

- Models: Corset, Lantern, BoomBox
- Source: https://github.com/KhronosGroup/glTF-Sample-Assets
- Model assets: CC0-1.0
- Metadata files: CC-BY-4.0
- Distribution: source assets, textures, latent exports and rendered images are not committed to the public repository

## Other Python packages

Python packages listed in `pyproject.toml`, `requirements-cpu.txt` and `requirements-gpu.txt` are installed from their upstream distributions and retain their respective licenses.
