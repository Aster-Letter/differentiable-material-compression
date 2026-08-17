# SciFiHelmet monotone BaseColor curve v1

状态：冻结边界实验

正式配置：`configs/train/scifihelmet_monotone_basecolor_v1.yaml`
正式输出：`outputs/scifihelmet_monotone_basecolor_curve_v7/`

## 目的与边界

该实验从 SciFiHelmet 的 raw-PCA `4→7` affine parent 出发，在不改变一次 RGBA8 采样与单个 affine decoder 的前提下，尝试用单调约束控制 BaseColor 目标。控制器把 RGB mean/tail 与 opponent relative mean/macro 组合成预设曲线，每 250 step 审计一次；候选只有同时满足 composite 进度和各项 guard 才能推进，否则降低步幅、学习率或投影回最近接受状态。

本实验只使用 31 个 train camera × 6 个 light，不访问 formal holdout，不连接 SCOW，不执行 UE 验收，也不改变正式部署主线。它回答的是“单调 aggregate 约束能否安全地沿渲染目标推进”，不是寻找课程最终 winner。

## 复现

所需 ignored parent 为：

```text
outputs/scifihelmet_c4_affine_v1/preflight/a874ad-r2/p0-raw
artifact_hash = 07c59e9cc07a00f5b0d4740f1c64d46db56d5684dd0ea9513d1f9a94b2c97799
```

运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_scifihelmet_monotone_basecolor.py preflight
.\.venv\Scripts\python.exe scripts/run_scifihelmet_monotone_basecolor.py train
.\.venv\Scripts\python.exe scripts/run_scifihelmet_monotone_basecolor.py report
```

也可用 `all` 串行执行三阶段。配置缺少、parent 越界或 artifact hash 不匹配时必须 fail closed。

## v7 结果

控制器在 step `1250` 收敛，并在下一目标 step `1500` 无法找到同时通过所有 guard 的更新，因此接受 1250 为 endpoint。证书通过，`formal_holdout_accessed=false`。

| 指标 | v7 endpoint |
|---|---:|
| mean HDR MAE | 0.0149107 |
| worst HDR MAE | 0.0547428 |
| mean display SSIM | 0.888977 |
| BaseColor Q8 MAE | 0.0329367 |
| generic chroma retention | 1.47007 |
| mean normal angle | 3.87043° |
| roughness MAE | 0.0775178 |
| metallic MAE | 0.0774750 |

相对严格 `S-separated@10k`，v7 的 mean HDR MAE 改善约 `4.18%`，但它不再 byte-exact，约 `99.77%` 的 chromatic texel 丢失至少 25% 的自身 chroma。相对 unconstrained `U0@10k`，v7 的 mean HDR MAE 仍回退约 `26.1%`。因此 aggregate chroma retention 大于 1 不能证明局部颜色忠实，单调 controller 也不能消除 rank-4 表示中的容量分配取舍。

## 课程报告用结论

- Exact BaseColor 说明严格保色会显著挤占 normal、roughness、metallic 和渲染质量预算。
- Monotone curve 说明连续、逐步、带 guard 的多指标控制仍可能放大总体色度却破坏大量具体纹素颜色。
- 两者共同支持“SciFiHelmet 的主要问题是单个 global rank-4 子空间的资产相关容量冲突”，不支持继续在同一 objective 上无界增加训练步数。

正式报告使用 v7 的 `final_summary.json` 和 `render_comparison_4view.png`；v1–v6 只作为控制器开发记录，不进入主结论。
