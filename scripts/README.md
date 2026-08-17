# Scripts

本目录保存可复现的命令入口；实验数学与可复用实现应位于 `src/cg_frontier/`。脚本不得依赖未记录的绝对路径，也不得把 formal holdout、凭据或现场下载写进正式训练路径。

## 生命周期

- **Active mainline**：`train_c4_render_ablation_20k.py`、Lantern 40k/160k continuation、对应 analyze/export/verify 脚本。
- **Shared workflow**：环境检查、资产预处理、参考渲染、结果分析、UE 导出与 manifest/hash 验证入口。
- **SCOW adapters**：`scow_*`、`remote_run_*`、`build_*_bundle.py`。本地打包，远端无网络运行，用户手动上传与提交。
- **Frozen reproduction**：repair、interpolation、filter-aware、Hybrid、ARC、DTF、旧 C4 affine、BaseColor-priority、Exact BaseColor 与 monotone 入口。它们保留历史协议和负结果，不作为新实验默认模板。
- **Sealed local-only**：由根 `.gitignore` 明列的封存评估入口。普通整理、测试收集和公开文档不得读取或纳入。

## 常用入口

- `check_environment.ps1`：CUDA/PyTorch/nvdiffrast/Blender 主环境验收。
- `check_ue_project.ps1`：UE 版本、RHI、关键渲染设置、忽略规则和最近启动日志检查。
- `train_c4_render_ablation_20k.py`：三模型同起点、两臂 fresh 20k 主实验。
- `continue_c4_render_ablation_lantern_40k.py`：Lantern 两臂 20k→40k 精确续训。
- `continue_c4_render_ablation_lantern_render_160k.py`：Lantern `material_render` 40k→160k 精确续训。
- `run_scifihelmet_exact_basecolor.py`：严格 BaseColor 的 audit/preflight/train/report 边界实验。
- `run_scifihelmet_monotone_basecolor.py`：从 raw-PCA parent 出发的单调约束曲线实验；只用于冻结复现。
- `build_course_report_materials.py`：从已验证 ignored 输出构建课程图片胶囊和 SHA-256 manifest。
- `cleanup_course_handoff_artifacts.py`：先 `prepare` 逐文件固化 hash，再 `apply` 执行严格 manifest 驱动清理。

完整阶段关系见 `docs/PROJECT_HISTORY_20260815.md`；远端操作合同见三个 `docs/SCOW_C4_*.md` guide。
