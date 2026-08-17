# Tests

优先测试跨工具最容易出错的契约：颜色空间、glTF metallic/roughness 通道、normal Y/TBN、decoder 导出等价性、量化前向、checkpoint 续跑和固定相机渲染回归。

- `test_c4_render_ablation*.py`：20k 配对实验、Lantern 40k/160k exact resume、采样 hash、观察节点、bundle 与 formal-holdout guard。
- `test_affine_*`：C4 single-affine 主线及其冻结诊断。
- `test_exact_basecolor*`、`test_ue_exact_basecolor_export.py`：严格保色 codec、lattice、checkpoint、运行时导出和 UE HLSL 合同。
- `test_monotone_basecolor.py`：单调曲线、multi-metric guard、trust-region、投影与状态恢复合同。
- Hybrid/ARC/DTF/repair 测试：历史复现契约，不能仅因非主线而删除。
- `gpu`、`ue`、`asset` marker：需要相应本地环境；默认 CPU 回归可用 `pytest -m "not gpu and not ue and not asset"`。

默认 pytest 临时根位于 ignored `.scratch/pytest/run-<uuid>`，避免在仓库根生成大量 `.pytest-*`。Windows 沙箱若在 session cleanup 报 ACL 错误，应在普通本机终端用唯一 `--basetemp` 复核，不把该环境错误解释为测试失败。
