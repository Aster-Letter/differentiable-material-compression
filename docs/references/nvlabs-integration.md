# NVlabs 集成调研

最后核对：2026-07-15T18:12:06+08:00

## nvdiffrast

- 官方仓库：https://github.com/NVlabs/nvdiffrast
- 当前 README 给出的安装方式为 `pip install ... --no-build-isolation`，需要 `setuptools`、`wheel`、`ninja` 和本机 CUDA/C++ 编译工具链。
- 适合作为现代训练环境中的直接依赖，使用 CUDA rasterizer context 支持本机和无显示服务器。
- 许可证为 NVIDIA Source Code License；固定来源 commit，并保留许可证与引用。

## nvdiffmodeling

- 官方仓库：https://github.com/NVlabs/nvdiffmodeling
- 仓库仅 10 个提交，最新 release 为 2021-04-13；README 测试环境为 Python 3.6、PyTorch 1.8。
- 包含本项目需要参考的 OBJ/material/texture、GBuffer 属性插值、GGX/PBR、相机与优化循环，但原任务是外观驱动模型简化，不是贴图通道压缩。
- 不适合作为本项目可变主仓库或锁定旧环境的理由；应固定 commit 作为只读参考，选择性迁移概念与必要函数，并为跨版本行为补测试。

## 推荐集成

1. 项目主代码使用现代 Conda/PyTorch/nvdiffrast 环境。
2. `third_party/` 中固定 nvdiffmodeling 来源 commit，只读保留，不直接在其目录开发。
3. 主代码按本项目语义实现资产 I/O、PBR、codec、训练、评测与导出；参考/迁移 NVlabs 的必要函数时保留出处和许可证说明。
4. 先复现 nvdiffrast 官方最小样例，再复现一个固定相机的 PBR 参考图；不以旧 `train.py` 整体跑通作为里程碑。
5. 对颜色空间、TBN/normal、roughness/metallic、GGX 参数和导出 decoder 做契约测试，防止“能运行但语义变化”。
