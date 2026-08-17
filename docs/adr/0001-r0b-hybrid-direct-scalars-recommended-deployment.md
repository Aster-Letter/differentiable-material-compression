# ADR-0001 草稿：SciFiHelmet R0b Hybrid direct-scalar 部署提案（已撤回）

- 状态：Withdrawn — 用户纠偏时未完成接受，不是有效架构决策
- 日期：2026-08-04
- 范围：SciFiHelmet、NoMipmaps、当前 UE 5.8/D3D12 部署条件
- 来源任务：`tsk:5fe2ba`

## 决策

本文是在用户纠偏到达前已写入的中断草稿，已撤回，不得作为 accepted ADR 或课程最终压缩架构引用。R0b 现仅定位为 **filter-safe hybrid control / 画质安全上界**：其运行时从同一 UV 以 Wrap/bilinear 采样两张 UNORM8 纹理，BaseColor/roughness/metallic 直接过滤，只有 normal 经 `2→6→2` decoder。它证明了解决主要插值暗斑的可行上界，但两次采样和 32 MiB UE actual resident 不满足“actual resident < 32 MiB 且 filter-safe”的强目标。

同时保留 **frozen pre-QAT hard `4→8→7`** 为 legacy single-sample comparator，不覆盖、不删除，也不将其资产路径改名为 R0b。Core-4 原始材质继续是质量 reference，不是任一压缩架构。

## 背景

legacy single-sample 表示的真实部署路径是 `bilinear latent → nonlinear decoder`。这一非线性过滤在 texel 之间产生了训练 texel-center 未直接覆盖的插值暗斑，尤其出现在 D2 黄管和 D3 灰板。既有单采样修复、Hybrid C1、factorization 与 D7-P 将问题收敛到 roughness strict non-regression；R0b 改为 BaseColor/roughness/metallic 直接过滤，只保留 normal-only decoder。

R0b 通过 13/13 offline gate；normal P95 为 `0.929772°`，roughness/metallic texel MAE 均为 `0`，repair-selection HDR MAE 为 `0.000603383`，display SSIM 为 `0.999569873`。UE 机器验收确认双 sampler/HLSL 语义、逐像素无损读回、Map Check 0 error；D2/D3 相对 legacy baseline 的 reference-MAE 分别改善 `23.31%/24.10%`，D1 为 `-0.07%`，只视为持平。用户人工近景复核未见突兀瑕疵。

## 成本与记账

| 角色 | 采样 | decoder | 参数 / 权重 | MAC/像素 | logical raw | UE actual resident |
|---|---:|---|---:|---:|---:|---:|
| Core-4 reference | UE 现有实现为 3 次源纹理采样 | 无 | 0 / 0 B | 0 | 项目 raw comparator 32 MiB | 本轮未作统一实测 |
| legacy single-sample baseline | 1 | `4→8→7` | 103 / 412 B | 88 | 16 MiB | 16 MiB |
| R0b filter-safe control | 2 | normal-only `2→6→2` | 32 / 128 B | 24 | 28 MiB | 32 MiB |

R0b 的 24 MAC 只是 decoder 静态计数，不是 frame-time 或完整 shader 性能实测。R0b 两张纹理在 UE 中都为 `PF_B8G8R8A8`、各 16 MiB，因此 actual resident 为 32 MiB；不得用 28 MiB logical raw 宣称实际显存压缩。相对项目的 Core-4 32 MiB raw 比较预算，当前没有实际显存压缩收益。

## 后果与限制

- 收益是解决旧 nonlinear latent filtering 的主要插值暗斑，并降低 normal decoder 成本；不是当前 UE 格式下的实际显存压缩。
- R0b 需要两次纹理采样，高于 legacy baseline 的一次采样。
- D1 未体现明确视觉改善，不宣称所有局部都优于 legacy baseline。
- 本决策只覆盖 SciFiHelmet、NoMipmaps、当前双线性 Wrap、相机/灯光和 UE 部署条件。不包含 mip/BC、AO、RenderDoc、完整 GPU benchmark、多资产泛化或报告排版。
- 当前 `MaterialLab*` 人工查看的未保存状态不是冻结资产，不得写回或覆盖既有地图。

## 验证与复现入口

- Release：`outputs/release/scifihelmet/r0b_v1/release_manifest.json`
- Claim matrix：`outputs/release/scifihelmet/r0b_v1/claim_matrix.json`
- Reproduction report：`outputs/release/scifihelmet/r0b_v1/reproduction_report.json`
- Offline summary：`outputs/compression/scifihelmet/hybrid_direct_scalars_v1/final_summary.json`
- UE acceptance：`outputs/deployment/scifihelmet/hybrid_direct_scalars_r0b/ue_evidence/ue_acceptance_summary.json`

冻结输入在两个独立输出根运行既有 exporter；manifest、HLSL、PNG、NPZ 与 fixed probes 逐字节/SHA-256 一致。定向回归为 `8 passed`，相关 6 个 Python 文件通过 `py_compile`，未读取 formal holdout，未启动 GPU 训练或 UE。
