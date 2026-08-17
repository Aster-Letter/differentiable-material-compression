# Windows 本机环境

最后更新：2026-08-06

## 已验证组合

- Windows + RTX 4060 Laptop 8 GB，NVIDIA 驱动 561.17
- Visual Studio 2022 Build Tools 17.14，Desktop C++ workload
- CUDA Toolkit 12.6，`nvcc 12.6.85`
- Miniforge 环境 `cg-frontier`，Python 3.11
- PyTorch 2.12.0+cu126
- nvdiffrast 0.4.0，固定提交 `253ac4f`
- Blender 5.2.0 LTS
- Epic Games Launcher 20.1.0
- Unreal Engine 5.8.0，Launcher build `55116800`
- UE Blueprint 项目 `ue_demo/CGCompressionDemo/CGCompressionDemo.uproject`

## 工作区 `.venv` 恢复记录（2026-08-06）

本机在 Conda 未进入 PATH 时，已另行验证仓库根目录的忽略环境 `.venv`：

- Python 3.12.4
- PyTorch 2.12.0+cu126，CUDA runtime 12.6
- Visual Studio 2022 Build Tools 17.14.36
- CUDA Toolkit 12.6.85
- nvdiffrast 0.4.0，固定提交 `253ac4f`
- pytest 9.1.1

恢复命令：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install ninja
```

nvdiffrast 必须从 **Visual Studio 2022 x64 Native Tools** 环境编译，并设置：

```bat
set DISTUTILS_USE_SDK=1
set CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set MAX_JOBS=1
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt --no-build-isolation
```

本机同时安装了 Visual Studio 2026。其中文 `cl` 输出是 UTF-8，而 PyTorch 2.12 的 Windows compiler probe 固定按 OEM codec 解码，会先触发 `UnicodeDecodeError`；即使临时改为 UTF-8 decoder，CUDA 12.6/项目也没有验证 VS 2026。因此不要让 `vswhere -latest` 自动选中 VS 2026，固定使用 `%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`。若 VS 2022 的本地化 `cl` 也触发相同解码错误，可在仅用于构建的 `PYTHONPATH/sitecustomize.py` 中设置：

```python
import torch.utils.cpp_extension
torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8",)
```

该 hook 只用于原生扩展构建，不进入仓库运行时或训练环境。最终验收必须包含真实 CUDA 张量反向传播，以及：

```powershell
$env:CUDA_HOME="$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
.\.venv\Scripts\python.exe -m pytest tests/test_nvdiffrast_smoke.py -q -p no:cacheprovider
```

2026-08-06 的 smoke 结果为 `1 passed in 4.67s`。只通过 `import torch` 或 `import nvdiffrast` 不算环境完成。

验证入口：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_environment.ps1
```

验证必须包含 CUDA 张量计算，以及 `tests/test_nvdiffrast_smoke.py` 的光栅化、插值、纹理采样和反向传播；只通过 `import` 不算完成。

## 从仓库复现 Python 环境

```powershell
conda env create -f environment.yml
conda run -n cg-frontier python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
conda run -n cg-frontier python -m pip install wheel ninja
```

Windows 编译 nvdiffrast 前，先从开始菜单进入 **x64 Native Tools Command Prompt for VS 2022**，并显式设置两个变量：

```bat
set DISTUTILS_USE_SDK=1
set CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.6
conda run -n cg-frontier python -m pip install -r requirements-gpu.txt --no-build-isolation
```

缺少 `DISTUTILS_USE_SDK=1` 时，PyTorch 2.12 会拒绝构建以避免重复激活 VC 环境；这不是 CUDA 编译失败。

## Blender 与 MVP 资产

Blender 安装：

```powershell
winget install --id BlenderFoundation.Blender --exact
```

MVP 资产是 Khronos SciFiHelmet。来源与稀疏克隆命令见 [`../assets/README.md`](../assets/README.md)。本机已用 Blender headless 导入确认 1 mesh、1 material、4 images。

## Unreal Engine 项目

已完成：

1. 安装 UE 5.8.0；
2. 建立 Blueprint、Desktop、Maximum Quality、DX12/SM6 项目；
3. 验证编辑器实际选择 RTX 4060 与 D3D12 SM6；
4. 关闭硬件 ray tracing 与 Substrate，保留标准 deferred material/base pass；
5. 将嵌套项目的 DerivedDataCache、Intermediate、Saved 等目录排除在 Git 外；
6. 建立 `MaterialLab`，使用项目级 `Manual` 且关闭自动曝光，Post Process Volume 启用并设为无限范围；
7. 通过 Interchange `Assets` 管线导入 SciFiHelmet，保留源法线/切线、关闭 Nanite，并验证 1 个静态网格体、1 个材质实例和 4 张纹理；
8. 建立 `M_SciFiHelmet_Reference`，用 BaseColor、Normal、MetallicRoughness.G/B 手工复现 Core-4 材质。

项目检查：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_ue_project.ps1
```

材质与资产契约见 [`ASSET_CONVENTIONS.md`](ASSET_CONVENTIONS.md)。

内容路径已通过 UE 内容浏览器整理并修复重定向器：`/Game/Imported/` 保存大型、可重建资产且不进入 Git；`/Game/CGCompression/` 保存地图、手写材质和蓝图并进入 Git。磁盘 `Content/`、内容浏览器“内容”和 UE 虚拟路径 `/Game/` 表示同一个挂载根，项目中不要另建名为 `Game` 的文件夹。

## 安装事件记录

初始化时 winget 的 CUDA 12.6.3 本地安装器仍在后台工作，又误启动了 CUDA 12.6.0 网络安装器。后者正确报告“其他安装程序正在运行”，随后被关闭；最终只保留 12.6.3，并以实际 `bin/include/lib` 写入、安装进程退出和 `nvcc` 验证为完成依据。以后不要因外层命令暂无输出而重复启动 CUDA 安装器。
