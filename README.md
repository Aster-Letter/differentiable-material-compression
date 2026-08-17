# Differentiable Material Compression

面向实时 PBR 部署的可微材质压缩研究实现。项目以 SciFiHelmet Core-4 材质为主案例，研究单张 RGBA8 潜在贴图、一次纹理采样和轻量解码器在 GPU 双线性过滤下的最终渲染质量。

> **English summary**
>
> This repository studies differentiable compression of PBR texture channels for real-time rendering. It combines PCA initialization, a single-sample global affine decoder, render-aware optimization, and controlled Unreal Engine validation. The current evidence covers three test assets, one random seed, and isolated single-object UE scenes, so the reported gains should not be read as a cross-asset guarantee.

**Keywords**　differentiable rendering · PBR · texture compression · PCA · Unreal Engine

当前部署主线将 decoder 固定为全局 `4→7` affine，使线性过滤与解码可交换，并把运行时成本压到 28 MAC。质量目标以固定相机和光照下的最终渲染外观为主；BaseColor、normal、roughness、metallic 和颜色指标用于诊断容量分配与明显伪影。

当前实验阶段已经从 SciFiHelmet 单资产诊断推进到多资产渲染监督对照：

- `Corset / Lantern / BoomBox` 以同一个 `raw_q4` PCA 起点，完成 `material_only` 与 `material_render` 两臂配对 20k；
- Lantern 两臂续到 40k，随后仅将 `material_render` 精确续到 160k；
- 渲染监督的收益具有资产与训练阶段依赖性。Lantern 在 40k 后继续改善 mean audit HDR，但 160k 仍以少量 worst-case、normal、metallic 与色度回退换取平均渲染和 roughness 改善，因此当前结论是 Pareto 取舍，而不是统一 winner。

项目从立项到冻结版本的里程碑、负结果与证据入口见 [Project history](docs/PROJECT_HISTORY_20260815.md)。公开数字和结论边界集中在 [Results and limitations](docs/RESULTS.md)。

## 预算与研究边界

- 潜在贴图：单张 2048² RGBA8，16 MiB actual resident；
- 运行时：一次 bilinear texture sample；
- decoder：单个 unconstrained `4→7` affine，35 parameters、140 B、28 MAC；
- 材质：linear BaseColor、切线空间 normal、roughness、metallic；AO 不在 Core-4 目标中；
- Unreal Engine 资产二进制、训练输出、模型、原始资产和内部协作记录不进入 Git。

历史 full-cube safe 参数化和非线性 filter-aware decoder 均保留为有效对照，但不再作为新实验默认。full-cube safety 在多资产上造成明显质量损失；后续 raw-affine lineage 使用部署一致的有界输出后处理，并单独记录其指令预算。

## 快速开始

要求 Python 3.11。CPU 测试不需要 Unreal Engine 或本地资产：

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -m pytest -m "not gpu and not ue and not asset"
```

获取 CC0 SciFiHelmet：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fetch_scifihelmet.ps1
```

CUDA/nvdiffrast 与 Windows/UE 设置见 [Windows setup](docs/SETUP_WINDOWS.md)。资产约定见 [Asset conventions](docs/ASSET_CONVENTIONS.md)。

SCOW 三模型 20k 与 Lantern 续训入口分别见 [20k guide](docs/SCOW_C4_RENDER_ABLATION_20K_GUIDE.md)、[40k guide](docs/SCOW_C4_RENDER_ABLATION_LANTERN_40K_GUIDE.md) 和 [160k guide](docs/SCOW_C4_RENDER_ABLATION_LANTERN_RENDER_160K_GUIDE.md)。

## 冻结版本

`v0.1.0` 保存项目冻结时的研究实现。代码和消敏文档继续留在公开仓库，大型训练产物、原始资产和 UE 二进制进入私有冷归档。这个版本不再追加训练结果，也不改变已经记录的实验合同。

## 仓库边界

公开仓库只包含源码、配置、测试、正式文档和经过消敏的小型结果摘要。内部 agent/session、Obsidian 工作区、formal holdout 数据、用户截图、checkpoint、latent 与 UE `.uasset/.umap` 均保持本地。详见 [Public repository policy](docs/PUBLIC_REPOSITORY.md)。

## License

本项目源码采用 MIT License。第三方依赖、参考实现和 SciFiHelmet 资产分别服从其原始许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
