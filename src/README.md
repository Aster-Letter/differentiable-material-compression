# Source

`src/cg_frontier/` 保存可复用实现，顶层 CLI 只负责装配配置与输入输出。第三方源码不直接混入本目录。

- `assets/`：glTF/Core-4 读取、材质语义、确定性 mesh 合并与资产验证。
- `render/`：GBuffer、相机/灯光 rig、PBR、canonical/decode-then-filter 渲染路径。
- `compression/`：材质表示、PCA/affine、历史 Hybrid/ARC/DTF、Exact/monotone BaseColor，以及当前 `render_ablation*` 训练与续跑合同。
- `experiment_io.py`：实验无关的结果树清单、流式 SHA-256、确定性 JSON+sidecar、有限值树与 campaign-root 路径保护。

当前部署主线是单张 RGBA8、一次硬件过滤采样和单个 `4→7` affine。历史 nonlinear、Hybrid、decode-then-filter、Exact BaseColor 与 monotone 模块属于 frozen reproduction/quality controls；共享模块不得静默改变它们的采样、量化或恢复语义。
