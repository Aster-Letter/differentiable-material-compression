# Assets

主资产为 Khronos `SciFiHelmet`（CC0）：

https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/SciFiHelmet

下载的原始文件放在 `assets/source/`，Blender/脚本处理结果放在 `assets/processed/`；两者默认不进入 Git。每份处理结果必须能由脚本或操作记录从来源重建，并保留来源、许可证、哈希、通道与颜色空间说明。

当前本机使用稀疏克隆：

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/KhronosGroup/glTF-Sample-Assets.git assets/source/glTF-Sample-Assets
git -C assets/source/glTF-Sample-Assets sparse-checkout set Models/SciFiHelmet
```

入口文件为 `assets/source/glTF-Sample-Assets/Models/SciFiHelmet/glTF/SciFiHelmet.gltf`。Blender 5.2 冒烟导入结果为 2 objects、1 mesh、1 material、4 images。

## 简单非金属候选

2026-08-09 按“先简单、优先非金属”的教师建议新增本地候选下载。筛选、分级、通道审计与来源见
`docs/SIMPLE_NONMETAL_ASSET_CANDIDATES_20260809.md`，逐文件哈希见
`assets/source/SIMPLE_NONMETAL_DOWNLOAD_MANIFEST_20260809.md`。

- Khronos Core glTF/CC0：`assets/source/simple_nonmetal_khronos/`
- Poly Haven glTF/CC0：`assets/source/simple_nonmetal_polyhaven/`

上述下载目录仍受 `/assets/source/` ignore 规则保护，不进入 Git；正式实验前必须为每个资产另建显式的通道、坐标、切线和 valid-UV 契约，不能直接继承 SciFiHelmet 的 identity-node 假设。
