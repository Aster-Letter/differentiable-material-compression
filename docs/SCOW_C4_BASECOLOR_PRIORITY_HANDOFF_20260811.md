# SCOW C4 BaseColor Priority Handoff

## Current Gate Status

Helmet Gate 1 has completed remotely. The implementation is recorded by
`4d6620b` and the self-contained summary fix by `6105d3e` on branch
`aster/c4-affine-mainline`.

- Phase 0 Job `36078` selected Corset and BoomBox.
- Gradient audit Job `36134` completed after diagnostic/preflight fixes.
- N0-control Job `36139`, BC80 Job `36140`, and BC90 Job `36141` each completed
  fresh 10k training with 1k/5k/10k checkpoints.
- Fixed summary preflight Job `36154` and formal summary Job `36155` completed;
  the summary contains six PNG figures and a verified manifest.
- Archive Job `36160` completed with 99 payload files, size `998,191,225` bytes,
  and remote SHA-256 prefix `df03f065...dabf2`.

The remaining recovery step is local, not training: download the archive,
recompute its full SHA-256, reject unsafe tar paths, validate every internal
manifest entry, and reload all nine checkpoints. Until that succeeds, report
the archive as remotely verified but not yet recovered outside SCOW.

After local recovery, the next conversation must present Gate 1 visual and
numeric Pareto evidence and wait for the user to choose BC80 or BC90 as B*.
It must not infer a winner or start Phase 2 automatically.

## Scope

This handoff records the approved C4 BaseColor-priority campaign for the SCOW
management task `tsk:a26131` / session `ses:a4efef`.

It does not itself authorize `sbatch`. Start jobs only after the user explicitly
instructs the SCOW management conversation to run this campaign.

Frozen runtime contract:

- 2048x2048 linear RGBA8 latent texture;
- one filtered texture sample;
- one unconstrained 4-to-7 affine decoder;
- scalar saturate and Normal XY disk projection;
- optional square-chroma compander;
- analytic decoder budget at most 80 scalar-instruction equivalents;
- no network, extra texture, extra sample, extra latent channel, UE, or formal holdout.

## Transfer

Use the checked transfer bundle and its sidecar manifest under:

`transfers/outgoing/c4-basecolor-priority-20260811-v2/`

Extract only into `$HOME/projects/cg_frontier`, verify the archive SHA-256 and
every manifest entry first, and do not overwrite unrelated remote projects.

The bundle contains the three CC0 complex candidates, the SciFiHelmet source
and processed Core-4 inputs, task source/config/scripts/tests, and the source
download manifest. It contains no historical outputs, formal holdout, UE
Config, credentials, logs, checkpoints, or local caches.

## Required Order

1. Verify an empty `squeue -u "$USER"`, the RTX5090 GRES, the extracted file
   manifest, and the existing SCOW Python/CUDA environment.
2. Run one RTX5090 10-step preflight for each new executable mode before its
   first formal use. Keep one PENDING/RUNNING job at a time.
3. Phase 0: run `asset-screen`. It must produce a valid `summary.json` and select
   two different validators. If fewer than two pass, stop the campaign.
4. Phase 1: run `helmet-audit`, then fresh `N0-control`, `BC80`, and `BC90`, each
   to 10k with 1k/5k/10k checkpoints. Generate the Helmet summary figures.
5. Return Gate 1 metrics and anonymous figures to this conversation. Do not run
   a compander candidate until the user selects BC80 or BC90 as B*.
6. Phase 2 after Gate 1 authorization: run the 1000-step post-hoc oracle from
   B*@10k, then fresh B*+compander to 10k with `--allow-compander`. Generate the
   updated summary and return Gate 2 evidence. Do not start complex training
   until the user accepts Gate 2.
7. Phase 3 after Gate 2 authorization: for each Phase 0 selected asset, run its
   own gradient audit, then fresh `N0-control`, B*, and B*+compander to 10k.
   Pass `--allow-phase3` through the wrapper and `--allow-compander` only for the
   selected compander candidate. Generate one complex summary per asset.
8. Stop after the multi-asset figures, numeric Pareto report, verified archive,
   and result recovery. Do not enter UE, 30k/40k, formal holdout, commit, or push.

## Slurm Contract

- partition `Students`;
- one node, one `RTX5090`, four CPUs;
- preflight at most 30 minutes, formal job at most four hours;
- omit memory, account, and QOS;
- logs in `logs/slurm/c4-basecolor-priority/%x.%j.{out,err}`;
- results in `outputs/remote/c4-basecolor-priority/<job-id>/`;
- each submission must pass the strict empty-queue guard.

Use:

```bash
bash scripts/scow_submit_c4_basecolor_priority.sh preflight <mode> [mode args]
bash scripts/scow_submit_c4_basecolor_priority.sh formal <mode> [mode args]
```

The wrappers refuse a login node and require `SLURM_JOB_ID`. The valid modes are
`asset-screen`, `helmet-audit`, `helmet-candidate`, `helmet-oracle`,
`helmet-summary`, `complex-audit`, `complex-candidate`, and `complex-summary`.

## Return Contract

After every job, append a concise status entry to `tsk:a26131` and `ses:a4efef`.
At each gate and at campaign completion, send the following back to this
conversation:

- job ID, mode, state, exit code, and elapsed time;
- GPU name/memory, Torch/CUDA version, and environment path;
- exact command, config/source hashes, and queue state;
- stdout/stderr and job-scoped output paths;
- checkpoint names, sizes, SHA-256 values, and reload result;
- gradient-audit lambda and both achieved BaseColor shares;
- finite/safety/compander checks and 1k/5k/10k metric trajectory;
- generated technical/anonymous figure manifests and hashes;
- completion, stop-gate, or failure conclusion;
- files recovered outside SCOW and anything not yet recovered;
- remote cleanup candidates, without deleting them unless separately authorized.

Never infer a scientific winner from a weighted total score. Report the numeric
and visual Pareto evidence and wait for the user's gate decision.
