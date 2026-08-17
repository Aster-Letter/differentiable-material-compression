# Unreal Engine Demo

UE 是三小时时盒学习集成，不是课程核心依赖。目标是导入 SciFiHelmet 和潜在贴图，在 Material/Custom HLSL 中执行 tiny decoder，将结果接入 Base Color、Normal、Roughness、Metallic，并记录原始/压缩对比与 Material Stats。

`Binaries/`、`DerivedDataCache/`、`Intermediate/`、`Saved/` 等生成目录不进入 Git。

## 当前项目

- 项目：`CGCompressionDemo/CGCompressionDemo.uproject`
- 引擎：Unreal Engine 5.8.0，Launcher build `55116800`
- 类型：Blueprint、Desktop、Maximum Quality、DX12、Shader Model 6
- 渲染边界：标准 deferred material/base pass；硬件 ray tracing 与 Substrate 关闭
- 暂时保留：Lumen GI/reflections、Virtual Shadow Maps；正式对比关卡中固定曝光和灯光

初始模板打开的 Engine OpenWorld 地图只用于安装验证。当前已建立 `MaterialLab`，关闭自动曝光并导入 SciFiHelmet；手工 `M_SciFiHelmet_Reference` 明确连接 BaseColor、Normal、MetallicRoughness.G→Roughness 与 B→Metallic，AO 不进入首版 Core-4 真值。

内容约定：

- `/Game/Imported/SciFiHelmet/`：从 Khronos glTF 重建的大型导入资产，不进 Git；
- `/Game/CGCompression/Maps/`：实验地图，进入 Git；
- `/Game/CGCompression/Materials/`：原始材质、latent decoder 与对比材质，进入 Git；
- `/Game/CGCompression/Blueprints/`：相机/灯光切换与展示逻辑，进入 Git。

内容已通过 UE 内容浏览器整理到上述目标目录并修复重定向器。Git 检查确认 `/Game/Imported/` 被忽略，`/Game/CGCompression/` 可跟踪；不要在 Windows 文件系统中直接移动 `.uasset`。

运行项目级只读检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_ue_project.ps1
```
