# Public repository policy

## Included

- Python/PowerShell source, deterministic configs and tests;
- Unreal project/configuration and source-form `Content/Python` helpers;
- formal architecture, asset-contract and reproducibility documentation;
- small sanitized result summaries that contain no machine paths or private test cases.

## Local only

- `AGENTS.md`, `.agents/`, `docs-agent/` and Obsidian project state;
- course-delivery sources, report PDFs, submission packages, videos, student IDs, acknowledgements and submission contacts;
- `.private/`, credentials, signing private keys and local environment state;
- source/processed assets, training outputs, checkpoints, latent textures and raw logs;
- user screenshots and formal holdout cases/aggregates;
- Unreal `.uasset`, `.umap`, imported content, caches and Saved state.

The repository may be prepared while private, but changing its GitHub visibility to Public always requires a separate explicit approval after history, license, CI and evidence review.

## Reproducibility evidence

Public GPU/UE evidence is a sanitized JSON manifest containing the Git commit, dependency versions, input hashes, command identity, structural cost and selected metrics. It must not contain absolute paths. Large artifacts stay outside Git and are referenced only by hashes.

Valid evidence is accompanied by a SHA-256 manifest and a detached Ed25519 signature. The verification public key may be committed; the private key remains local.
