# Lantern C4 可微渲染 20k→40k SCOW 操作

本包是 `c4-render-ablation-20k-v1` campaign 的增量包，不复制约 800 MiB 的 Lantern 20k parent。远端必须继续保留并核验 Job `37477` 的原始结果；两个臂分别从各自 `step_20000/checkpoint.pt` 精确恢复 optimizer 与 RNG，串行训练到 40k。

## 上传、验证、应用

手动上传本地 `transfers/outgoing/c4-render-ablation-lantern-40k-v1.zip` 到：

```text
$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-40k-v1.zip
```

然后在 SCOW Shell 执行（把 `<ZIP_SHA256>` 替换为交付时给出的值）：

```bash
ROOT="$HOME/projects/cg_frontier/campaigns/c4-render-ablation-20k-v1"
ZIP="$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-40k-v1.zip"
PATCH="$HOME/projects/cg_frontier/transfers/incoming/c4-render-ablation-lantern-40k-v1"

echo "<ZIP_SHA256>  $ZIP" | sha256sum -c -
mkdir -p "$PATCH"
unzip -q "$ZIP" -d "$PATCH"

cd "$PATCH"
sha256sum -c LANTERN40K.MANIFEST.sha256
python3 payload/scripts/verify_c4_render_ablation_lantern_40k_bundle.py \
  --bundle-root "$PATCH" \
  --campaign-root "$ROOT"

cd "$ROOT"
sha256sum -c "$PATCH/PATCH_BASELINE.sha256"
cp -a "$PATCH/payload/." "$ROOT/"
sed 's#  payload/#  #' "$PATCH/LANTERN40K.MANIFEST.sha256" | sha256sum -c -
```

最后一条应逐文件显示 `OK`。验证器还必须输出：

```json
{"formal_holdout_present": false, "payload_files": 11, "preserved_parent_verified": true, "schema_version": 1, "source_job_id": "37477", "status": "bundle_verified"}
```

## Preflight

```bash
cd "$ROOT"
squeue -u "$USER"
bash scripts/scow_submit_c4_render_ablation_lantern_40k.sh preflight
```

记下 Job ID 为 `PF`。该作业对两个臂分别只续 10 步，并检查 forward/backward、optimizer、checkpoint reload、成对采样与父 lineage；上限 30 分钟，不代表预计耗时。

```bash
PF=37580
OUT="logs/slurm/c4-render-ablation-lantern-40k-v1/c4-ra40-preflight.${PF}.out"
ERR="logs/slurm/c4-render-ablation-lantern-40k-v1/c4-ra40-preflight.${PF}.err"
MARKER="outputs/remote/c4-render-ablation-lantern-40k-v1/${PF}/preflight_verified.json"

squeue -j "$PF" -o "%.18i %.28j %.10T %.10M %.10l %R"
tail -n 100 "$OUT"
[[ -s "$ERR" ]] && { echo "FAIL: stderr 非空"; cat "$ERR"; } || echo "OK: stderr 为空"
[[ -s "$MARKER" ]] && python3 -m json.tool "$MARKER" || echo "尚未通过或 marker 缺失"
```

只有队列中已无该作业、stderr 为空且 marker 的 status 为 `preflight_verified`，才提交正式作业。

## 正式 40k 续训

```bash
cd "$ROOT"
bash scripts/scow_submit_c4_render_ablation_lantern_40k.sh formal "$PF"
```

正式作业内部严格串行：`material_only 20k→40k`，再 `material_render 20k→40k`。观察节点为 25k/30k/35k/40k，30k/40k 保存完整 checkpoint；不启用提前停止。Slurm 上限为 4 小时。

```bash
JOB=37581
OUT="logs/slurm/c4-render-ablation-lantern-40k-v1/c4-lantern-ra40.${JOB}.out"
ERR="logs/slurm/c4-render-ablation-lantern-40k-v1/c4-lantern-ra40.${JOB}.err"
MARKER="outputs/remote/c4-render-ablation-lantern-40k-v1/${JOB}/formal_verified.json"

squeue -j "$JOB" -o "%.18i %.28j %.10T %.10M %.10l %R"
tail -n 100 "$OUT"
[[ -s "$ERR" ]] && { echo "FAIL: stderr 非空"; tail -n 120 "$ERR"; } || echo "OK: stderr 为空"
[[ -s "$MARKER" ]] && python3 -m json.tool "$MARKER" || echo "尚未通过或 marker 缺失"
```

结果根为 `outputs/remote/c4-render-ablation-lantern-40k-v1/<job-id>/Lantern/`。只有 `formal_verified.json` 存在且 result manifest 完整时，才将其视为有效 40k pair。

## 失败恢复

不要盲目原样重提。如果正式作业异常终止且目标臂已有完整 30k checkpoint，先检查日志确认不是输入/hash/环境错误，再执行：

```bash
bash scripts/scow_submit_c4_render_ablation_lantern_40k.sh resume \
  <material_only|material_render> <FAILED_JOB_ID> "$PF" yes
```

恢复入口只接受选定臂的 30k continuation checkpoint；会隔离 30k 之后的过期节点，并精确重放到 40k。如果另一臂尚未开始，会随后从其冻结 20k parent 正常续到 40k；如果另一臂已有不完整输出，则停止并要求单独诊断，避免覆盖证据。
