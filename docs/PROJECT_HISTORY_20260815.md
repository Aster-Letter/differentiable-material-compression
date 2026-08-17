# Project history and current state

状态日期：2026-08-15

分支：`aster/c4-affine-mainline`

本文是从项目建立到 Lantern 160k 实验结束的阶段性整理。它提供时间线和当前结论，不替代各实验 guide、ADR、结果 manifest 或 Git 历史。

## 结论先行

项目已经建立并验证了一条完整闭环：Core-4 glTF 材质读取 → 可微 GBuffer/PBR → 量化潜在贴图与实时解码器 → checkpoint/exact resume → SCOW 训练与归档 → Unreal Engine 同构导入和人工预览。

导师确认后的部署主线固定为一张 RGBA8、一次硬件过滤采样和一个全局 `4→7` affine（35 个 FP32 参数、28 MAC）。实验表明该极低成本表示并非在所有资产上同样有效：SciFiHelmet 暴露明显的全局 rank-4 容量/颜色冲突；Lantern 的 raw PCA 很差，但 learned affine 可大幅恢复；Corset 和 BoomBox 上渲染监督的增益则更混合。当前最合理的表述是“存在资产依赖的 Pareto 前沿”，而不是一个对所有材质都占优的 loss 或步数。

## 时间线

| 日期 | 阶段 | 主要产出与认识 |
|---|---|---|
| 2026-07-15 | 项目初始化 | 建立可微材质压缩代码、学习计划和 Python 环境骨架。 |
| 2026-07-17 | UE 演示起步 | 建立 Unreal Engine 项目与材质/GBuffer 前集成入口。 |
| 2026-08-02–03 | Core-4 边界 | 固定 BaseColor、切线 Normal、Roughness、Metallic 的颜色空间、通道、TBN 与量化约定；形成资产和 UE 验收边界。 |
| 2026-08-04 | nonlinear 单采样诊断 | legacy `4→8→7` 在 `bilinear latent → nonlinear decode` 处产生插值暗斑；R0b direct-scalar 证明 filter-safe 质量上界，ARC 验证激活区域约束，但二者不成为最终部署答案。 |
| 2026-08-05 | decode-then-filter | 建立四角解码后材质过滤的 C4/C5 DTF 质量主线。它提供高质量对照，但四次解码和隐藏层不满足后来确认的极低计算预算。 |
| 2026-08-06–08 | C4 single-affine | 导师明确四通道、一次采样、无隐藏层 `4→7` affine。完成 P0 PCA、L0/L1/L2、TV/cube、exact resume、UE 导出与 40k/80k 诊断。full-cube safety 和长训练未解决 SciFiHelmet 色度坍缩。 |
| 2026-08-08–11 | 颜色与采样因果线 | camera-relative light、通用色度 guard、BaseColor-priority、compander/gradient audit 逐项隔离。它们说明背光采样和 loss 权重会放大问题，但主要限制仍是单个 global rank-4 子空间的容量与目标冲突。 |
| 2026-08-12–13 | 多资产选择 | 从复杂金属候选中选定 Corset、Lantern，并以 BoomBox 作为较低难度对照；WaterBottle 因 80 个退化 UV 三角形排除。 |
| 2026-08-13–14 | 三模型 20k 配对实验 | 三模型从同一 `raw_q4` fresh 起点运行 `material_only` / `material_render`，固定 seed、采样 hash、24 train cameras、7 audit cameras、6 lights、256²。三个正式作业与归档全部通过。 |
| 2026-08-14 | Lantern 40k | 两臂从 20k 精确续到 40k；render arm 相对 20k mean audit HDR 改善约 17.56%，并在 40k 比 material-only 低约 2.36%。 |
| 2026-08-15 | Lantern render-only 160k | material-render 从 40k 精确续到 160k；mean audit HDR 再改善 4.79%，37/42 case 改善，display SSIM 增加约 0.000274，但 worst HDR 轻微回退 0.42%，normal/metallic/opponent/chroma 也有回退。结论为有用的 Pareto endpoint，不是全面支配。 |

对应 Git 主节点：`565c013 → 210538d → b7c25a9 → 876ba55/08ebac3/85a4ff6 → 8ff5e76 → 6d63167/f73272b/1e847b9 → 4d6620b → 3b5a12d/f3fb1f1/24d3e15`。

## 关键实验结论

### 1. 过滤顺序是第一类问题

非线性 decoder 与 bilinear filtering 不可交换。legacy 暗斑不是简单增加 texel-center loss 就能可靠消除；R0b 与 DTF 分别给出直接材质过滤和四角 decode-then-filter 的质量证据。该结论保留为历史设计依据，即使它们不满足当前部署预算。

### 2. 全局 C4 affine 是第二类问题

在一次采样、28 MAC 的合同下，PCA 与 learned 方法共享同一个运行时结构，差异只来自离线表示求解。SciFiHelmet 的 q6/cluster oracle 明显优于 global q4，而多轮训练会牺牲稀有颜色来改善平均材质/渲染指标；这支持“单个全局四维线性子空间容量不足”的解释。

### 3. 渲染监督不是普遍单调收益

20k 时，`material_render` 相对 `material_only`：

| 资产 | mean audit HDR | worst audit HDR | display SSIM | 解释 |
|---|---:|---:|---:|---|
| Corset | 回退 0.85% | 回退 30.14% | +0.000393 | 平均接近持平，case 分布和 worst 明显取舍 |
| Lantern | 回退 8.24% | 回退 0.76% | -0.000916 | 20k 尚未收敛到 render arm 优势 |
| BoomBox | 改善 16.09% | 改善 29.13% | -0.000397 | HDR 明显改善，但 SSIM/材质指标非统一改善 |

因此判断“渲染监督本身”的贡献必须使用两臂差值，并结合 case-level、worst 与材质诊断；不能用 train render loss 或加权总分直接选 winner。

### 4. Lantern 证明长训练有收益，但存在边际取舍

Lantern raw-q4 是明确难例。20k learned 两臂相对 raw PCA 的 mean audit HDR 分别改善约 63.69%（material-only）和 60.70%（material-render）。到 40k，render arm 反超 material-only；40k→160k 又获得平均渲染收益，但 worst-case 和色度不完全随之改善。后续若比较 40k/80k/120k/160k，必须为中间 checkpoint 重新跑同一 42-case audit；当前正式结果只支持 40k 与 160k endpoint 的完整 audit 结论。

## 当前可复现入口

- 运行时/架构权威：[ADR 0004](adr/0004-c4-single-affine-mainline.md) 与 [C4 experiment plan](C4_AFFINE_EXPERIMENT_PLAN.md)。
- 三模型 20k：[SCOW guide](SCOW_C4_RENDER_ABLATION_20K_GUIDE.md)，正式 jobs `37474 / 37477 / 37478`，归档 job `37489`。
- Lantern 40k：[SCOW guide](SCOW_C4_RENDER_ABLATION_LANTERN_40K_GUIDE.md)，正式 job `37581`。
- Lantern 160k：[SCOW guide](SCOW_C4_RENDER_ABLATION_LANTERN_RENDER_160K_GUIDE.md)，正式 job `37824`，归档 job `38202`。
- 本地 compact reports：`outputs/analysis/c4-render-ablation-*/report/`；训练产物和 UE 二进制保持 ignored/local-only。
- 产物保留与恢复：[Artifact retention policy](ARTIFACT_RETENTION_POLICY.md)。

## 当前停止点与下一步

1. 当前分支已包含 20k、40k、160k 的训练、验证、分析、SCOW 和 UE preview 源码；正式远端结果已经本地归档。
2. 160k 是 Pareto endpoint。若目标是选择 Lantern 的最佳停止点，下一步应对 80k/120k checkpoint 补跑只读 42-case audit，而不是继续无上限训练。
3. 若研究问题回到“渲染监督能否普遍优于材质监督”，需要增加资产与 seed；当前单 seed 三资产只能作为确定性配对案例，不支持方差或统计显著性声明。
4. 若目标是解决 SciFiHelmet 的明显颜色失败，应讨论放宽 global-q4/single-affine 契约，或把不可行性作为资产适用范围结论；不建议继续在同一 objective 上单纯加步数。
5. UE 当前用于 source/raw PCA/20k/40k/160k 人工对比。所有 `.uasset/.umap`、Imported、用户 Config 与 formal holdout 继续保持本地边界。
