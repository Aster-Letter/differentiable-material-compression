# 结果与适用边界

本页汇总公开仓库能够支持的主要观察。数字来自冻结配置、单随机种子和已经校验的结果清单。原始 checkpoint、纹理、UE trace 与截图留在私有冷归档，公开仓库保留复现入口和消敏后的汇总数据。

## 固定四通道预算

三项测试资产都从各自的 `raw_q4` PCA 表示开始，再优化同一张四通道潜在贴图和全局 `4→7` 仿射解码器。训练 20k 步以后，material-render 相对 PCA 的平均 HDR MAE 在 Corset、Lantern 和 BoomBox 上分别下降 8.50%、60.70% 和 5.20%。

这组改善不能全部归到渲染监督。20k 端点中，material-render 相对 material-only 在 Corset 和 Lantern 上的平均 HDR MAE 分别高 0.85% 和 8.24%，只在 BoomBox 上低 16.09%。Lantern 的恢复主要来自可学习表示；BoomBox 才显示出这一训练预算下的渲染监督增益。

## Lantern 长训练

Lantern 的两条分支从 20k 精确续训到 40k。material-only 的平均审计 HDR MAE 从 0.004674 降至 0.004272，material-render 从 0.005060 降至 0.004171。后者在 40k 端点低 2.36%，同时带来轻微的 BaseColor 和七通道综合误差代价。

material-render 继续训练到 160k 后，平均审计 HDR MAE 相对 40k 再下降 4.79%，42 个审计条件中有 37 个改善。SSIM 增加 0.000274，粗糙度误差下降 28.48%，七通道综合误差下降 5.22%。同一端点的最差 HDR MAE 上升 0.42%，特定对立色区域误差上升 5.68%，颜色保持指标下降 1.15%。160k 因此是偏向平均渲染与粗糙度的 Pareto 端点。

## 容量与部署

SciFiHelmet 的固定 rank-4 表示暴露了更明显的容量边界。增加颜色保护约束能够改善部分颜色区域，却会把误差转移到 normal、roughness 或整体材质指标。更大的 DTF 解码器提供了质量上界，同时增加纹理读取和算术成本。公开实现保留这些失败与高成本对照，默认部署主线仍使用全局 affine decoder。

在 UE 5.8 的受控单对象场景中，RGBA8 潜在贴图能够实时解码。逻辑通道压缩本身没有降低 RGBA8 驻留量。Lantern 的 2048×2048、12 mip 潜变量改用 BC7 后，实际驻留量为 5.375 MiB，相对相同潜变量的 RGBA8 格式下降 74.9%，相对源 Core-4 三张贴图下降 60.0%。固定视图检查通过，代表性单对象 BasePass 测量没有观察到可分辨的实质变化。

## 结论范围

- 多资产配对实验只有一个随机种子。
- Lantern 160k 没有支配全部材质和最差条件指标。
- UE 数字来自冻结的单对象隔离场景，不能外推到多实例、mip/LOD、流送压力或整帧性能。
- BC7 检查只覆盖 Lantern 和三个潜变量端点。
- formal holdout、原始训练产物和 UE 二进制不属于公开仓库内容。

复现入口见根目录 README、各 SCOW guide、`configs/`、`scripts/` 和 `ue_demo/CGCompressionDemo/Content/Python/`。资产通道、颜色空间和切线空间约定见 [Asset conventions](ASSET_CONVENTIONS.md)。
