# Configs

每次训练和评测使用独立 YAML 配置。至少记录资产、分辨率、latent 通道/格式、decoder、损失权重、相机/光照采样、seed、训练步数、checkpoint 与输出目录。

## 当前主线

- `train/c4_render_ablation_20k_v1.yaml`：Corset、Lantern、BoomBox 的 `raw_q4` 配对 20k。
- `train/c4_render_ablation_lantern_40k_v1.yaml`：两臂 20k→40k。
- `train/c4_render_ablation_lantern_render_160k_v1.yaml`：仅 `material_render` 40k→160k。

`c4_affine_*`、BaseColor-priority 与 camera/light 诊断配置保留为 C4 single-affine 的前序证据；repair、Hybrid、ARC、DTF 等配置为 frozen reproduction。历史配置保持原样，新的实验必须使用新文件名和新输出根，不在原配置中静默改权重、观察节点或恢复语义。

`train/scifihelmet_exact_basecolor_v1.yaml` 与 `train/scifihelmet_monotone_basecolor_v1.yaml` 是 SciFiHelmet 保色边界研究的 frozen reproduction。它们不替代多资产 C4 主线；所需原始资产与 P0-raw parent 保持 ignored，并由配置中的 artifact hash 校验。
