# C4 单仿射材质压缩：实施与实验计划

最后更新：2026-08-06

## 1. 要回答的问题

主问题：在 `RGBA8 + 一次过滤采样 + 4→7 affine` 的相同部署成本下，渲染感知 Learned Linear 是否稳定优于标准 PCA？

扩展问题：chart-aware TV 或 canonical cube render 是否能在不增加运行时 decoder 计算量的前提下改善 atlas 空洞、噪点或跨视角质量？这些方法只增加离线训练成本；最终仍折叠为同一个 35 参数 affine。

不在本轮回答：更高通道数、隐藏层网络、逐角 decode、动态专用 ROI loss、手工逐级 mip、formal holdout 上的调参，以及未经导师确认的“原生 GPU 指令数”结论。

## 2. 冻结契约

- 输入：每资产一张 2048² RGBA8、非 sRGB latent。
- 输出顺序：linear BaseColor RGB、tangent Normal XY、roughness、metallic。
- 运行时：一次普通 Texture Sample、28 affine MAC、Normal-Z 重建、一次 UE normal-Y bridge。
- 安全性：训练期按构造保证完整 `[0,1]^4` 域合法；部署只使用普通 `W,b`，不加条件投影或激活函数。
- mip：UE 线性平均生成、无 sharpening、派生 LOD、普通过滤和 Streaming。
- 数据边界：训练/选择数据与 formal holdout 隔离；formal holdout 不参与本计划执行。
- 资产边界：历史 DTF/R0b/ARC/legacy、outputs、Imported 与 `MaterialLab*` 不清理、不覆盖。

## 3. 候选矩阵

| ID | 初始化 | 基础材质/Helmet render | TV | Cube render | 是否改变运行时 |
|---|---|---:|---:|---:|---:|
| P0-safe | uniform-valid-texel PCA | 否 | 否 | 否 | 否 |
| L0 | P0-safe 深拷贝 | 是 | 否 | 否 | 否 |
| L1 | P0-safe 深拷贝 | 是 | 是 | 否 | 否 |
| L2 | P0-safe 深拷贝 | 是 | 否 | 是 | 否 |
| L3 | P0-safe 深拷贝 | 是 | 是 | 是 | 否；条件执行 |

P0-safe 是冻结 comparator。`L0@step0 == L1@step0 == L2@step0 == P0-safe` 只表示共同父状态，不是四个独立结果；三个 learned 候选必须深拷贝为互不共享可变 tensor/optimizer/RNG 的 child state。L0 至少完成一次 optimizer update 后才具有 Learned Linear 语义，正式比较使用 P0-safe 与共同 step-qualified endpoint。L2 的 cube RNG 独立，不能改变核心样本序列。L3 只有 L1/L2 至少一项显示可解释收益且没有不可接受的全局回退时才提出执行。

## 4. 分阶段实施

### M0：环境与保护栏

产物：可重建 Python/CUDA 环境、环境锁定记录、dirty ownership 记录、GPU/UE 进程检查命令。

验收：CPU 单元测试可运行；CUDA tensor test 可运行；需要渲染测试时 nvdiffrast forward/backward smoke 通过；没有读取 formal holdout；没有修改两份既有 UE 配置状态。

### M1：安全 affine 最小闭环

采用 TDD 逐条实现：

1. 训练参数化在 16 个 latent 角点和解析界上保证五个标量合法、Normal XY 位于单位圆内；
2. 训练前向与折叠后 `F.linear(z,W,b)` 数值一致；
3. Normal-Z 重建有限、正半球且单位长度，不执行额外 normalize；
4. affine 与 bilinear/线性 mip 平均交换；
5. manifest 固定记录 35 params、140 B、28 MAC、一个 resource/sample、输出语义与证书摘要。

停止条件：任何证书失败都先作为实现或数值故障诊断，不允许在 exporter 中偷偷缩放参数。

### M2：P0 与统一导出

在有效 UV 纹素上构造 7D 目标，标准化口径与逆变换必须进入 manifest。拟合 PCA 后，将四维表示与 decoder 校准到安全参数化，量化/反量化 RGBA8，并报告安全约束造成的额外误差。

验收：固定 seed/输入哈希下可复现；CPU 重载与导出回读一致；P0 使用与 Learned Linear 相同的 UE shader/cost；空白 atlas 不参与 PCA 拟合或指标归一化。

### M3：损失单元

L1：只在 valid、same-chart 的水平/垂直边计算 fake-quant latent Charbonnier TV，按有效边数归一。合成双岛测试必须证明不会跨 seam 平滑。

L2：六个 cube face 显式冻结 UV orientation/TBN；每面完整映射 `[0,1]²`，同时采样 atlas valid mask。六个 face-normal 正交相机与通用冻结光照共用一套 reference/decoded render 路径。合成方向纹理和法线测试必须捕获翻面、旋转、Y-sign 或 mask 错误。

权重：TV 与 cube 分别在 P0 做一次基础目标/扩展目标 latent-gradient 标定，目标比例写入配置与 calibration artifact，正式训练中冻结。比例数值在短预检后人工冻结，不能在运行中自适应。

### M4：配对 trainer 与精确恢复

实现 L0/L1/L2 独立输出目录和统一 runner。三者 manifest 都记录同一 `parent_p0_hash`，但 candidate/objective ID、optimizer/RNG 与可变 state 独立；P0-safe 本身没有继续前进的 optimizer。checkpoint 必须保存 latent、safe affine 原始参数、两类 optimizer、核心 RNG/sampler、cube RNG、step、phase、配置/输入哈希和 best 轨迹。

时间表：0–2k warmup，2k–35k joint main，35k–40k polish；保存 1k、5k、10k、20k、30k、35k、40k。共同续训只从完整状态恢复，保留 40k/80k/120k 每个 endpoint。

验收：中断前后下一批核心样本、loss 和参数更新在容差内一致；L0/L1/L2 的核心序列相同；只增加 cube RNG 不改变 L0/L1；不得按 wall time 提前截断某一候选。

### M5：正式训练前 timing

先做 CPU synthetic，再做 CUDA 10-step correctness，最后对 L0/L1/L2 分别做固定步数 timing。记录预热步数、测量步数、step time 分布、samples/s、峰值显存与各 loss 占比。若 L2 超出本机可接受预算，先调整离线 cube batch/frequency 的冻结配置并重做配对预检，不能改变运行时结构。

此阶段只估算 40k wall time，不据 28/1728 MAC 比例外推整条训练加速。

### M6：SciFiHelmet 40k 主实验

严格串行运行 P0、L0、L1、L2，确保任一时刻只有一个 GPU 训练进程。每个 endpoint 生成不可变 manifest、hash、曲线、材质/渲染指标、artifact 指标、证书与成本清单。

40k 后先并列报告，不做预设 winner。若三条 learned 曲线仍共同改善且额外 40k 的信息价值明确，则三者共同续到 80k；80k→120k 同理。120k 后如仍需 160k，重新记录决策依据。

### M7：UE 同构验收

在独立的新内容路径中创建一个共享 master material 和按候选/步数命名的 Material Instance；P0/L0/L1/L2 不产生 shader permutation。验证纹理回读、sRGB、mip generation、filter、Streaming、常量与 Y-sign。

同视角人工比较 PCA、Learned、历史质量对照和 reference。采集同一 UE 5.8 changelist / D3D12 SM6 下 representative/max PS/VS instructions、texture samples、samplers、HLSL hash、cooked texture bytes；actual resident 与 GPU time 另列环境依赖证据。

### M8：三资产泛化

先做许可证和技术审计，再冻结一项 metal-dominant、一项 nonmetal/rough、一项 high-frequency/complex-boundary 资产。三者只运行 P0、L0 与 SciFiHelmet 选出的扩展，不补做完整四因子矩阵。

验收：所有资产保持 Opaque/Core-4/2048²/单材质优先；不利结果完整保留。最终以四模型连续效果、case 级差异、UE 视觉和成本共同判断，而非用规划阈值自动改写结论。

## 5. 指标与报告

- 主要质量：paired linear HDR MAE。
- 共同解释：display SSIM、material composite、BaseColor、normal mean/P95、roughness、metallic、dark/halo/connected artifacts。
- 稳定性：完整曲线、endpoint 差分、证书 margin、seed/config/input/output hashes、恢复一致性。
- 离线成本：step time、samples/s、peak VRAM、cube 额外成本。
- 部署成本：解析 MAC/参数、UE compiler stats、samples/samplers、HLSL hash、cooked bytes、actual resident/runtime（后两者注明环境）。

规划期的 `3/4`、5% HDR、0.001 SSIM、10% material 等数值只作预期参考，不是自动 gate。先展示完整比较，再记录人工裁断与理由。

## 6. 可选拓展及触发证据

- Render-weighted PCA：当标准 P0 明显受低可见 atlas 区域支配，作为额外同成本基线。
- Edge-aware TV：L1 明显去噪但模糊真实边界时，新建 lineage。
- Reachable-set safety：全域安全预算被证明造成显著拟合损失时，只对哈希绑定的量化 latent+mip 集合做额外证书。
- Random/multi-seed：PCA-init 显示停滞或跨模型系统偏置时。
- Mip/footprint-aware training：UE 自动 mip 产生可复现的质量回退时。
- BC latent formats：需要比较真实 cooked/resident 成本时；不能从 RGBA 通道数直接推断收益。
- Plane/sphere/corner views：L2 cube 有收益但暴露几何偏置时。

每项拓展都必须有新配置、新输出目录和新 lineage；不在运行中的 L0/L1/L2 内改变 loss、权重或采样。

## 7. 当前恢复点

- 设计与实验矩阵已确认；L2 canonical cube 协议已确认。
- 生产实现和训练尚未开始。
- `tests/test_affine_material.py` 是暂停在环境建立前的第一个 RED 草稿，只描述 M1 的首个公共行为；尚无对应生产模块。
- 工作树中两份 UE Config 修改属于既有外部状态，保持不动。
- M0 环境已验收：`.venv` 使用 Python 3.12.4、PyTorch 2.12.0+cu126、CUDA Toolkit 12.6.85、VS 2022 Build Tools 17.14.36 与固定 nvdiffrast 0.4.0；真实 CUDA backward 和 nvdiffrast smoke 均通过。
- 历史 ARC/DTF/R0b/Obsidian 完成提交已 fast-forward 合入本地 `main`；当前分支为 `aster/c4-affine-mainline`。
- 下一动作是按 M1 逐条 TDD 推进；不会直接启动长训练。

## 8. 下一对话的具体 TDD 与短预检协议

下一对话的授权终点是完成 M1–M5 和短 CUDA preflight；不得自动进入 M6 的 40k 正式训练。每个编号先只写一个公共行为测试并确认 RED，再做最小 GREEN；RED 状态下禁止顺手重构。

### 8.1 模块与命名边界

- `src/cg_frontier/compression/affine_material.py`：safe-by-construction 参数化、折叠、直接语义 decode、证书和静态成本。
- `src/cg_frontier/compression/affine_pca.py`：标准 P0、latent 归一化、部署安全校准和 P0 artifact。
- `src/cg_frontier/compression/affine_regularizers.py`：chart-aware TV、梯度比例标定。
- `src/cg_frontier/render/canonical_cube.py`：冻结 cube geometry/UV/TBN/camera 与 cube loss 输入。
- `src/cg_frontier/compression/affine_training.py`：L0/L1/L2 配对状态、checkpoint、resume、timing；不修改 legacy `material.py` 或 DTF trainer 的语义。
- 测试分别放入 `test_affine_material.py`、`test_affine_pca.py`、`test_affine_regularizers.py`、`test_canonical_cube.py`、`test_affine_training.py`。正式配置与输出使用 `scifihelmet_c4_affine_v1` 独立命名。

### 8.2 M1 safe affine 测试顺序

| ID | RED 所描述的公共行为 | GREEN 验收 |
|---|---|---|
| A01 | 任意 raw 参数在 16 个 RGBA 角点上五个标量合法、Normal XY 在单位圆内，且折叠 affine 与训练前向一致 | float64 `atol/rtol=1e-12`；梯度 finite |
| A02 | 解析证书报告每个 scalar 的全域上下界、Normal 最大半径、margin、dtype 与 finite 状态 | 解析界包住 16 角点；NaN/Inf、错误 shape、非正 margin fail closed |
| A03 | 七个 affine 输出是直接材质语义，不经过 sigmoid/tanh/clamp/条件投影 | 拼接七通道逐值等于 `F.linear`；只在随后重建 Normal Z |
| A04 | Normal Z 在单位圆中心、近边界和合法随机点上有限、正半球且单位长度 | 不调用最终 normalize；仅允许 sqrt 的数值 epsilon guard |
| A05 | affine 与 bilinear/线性 mip 平均交换 | 比较的是七个 affine 通道，Normal Z 只在过滤后重建；不得误测 XYZ normal filtering commute |
| A06 | exporter/reloader 保留 35 个 FP32 `W,b`、输出顺序和证书 | 140 B、28 MAC、1 resource、1 filtered sample；reload 前后逐值一致 |

训练期采用以下全程平滑预算参数化，避免阈值触发修正：

- 每个 scalar row 在中心化 `u=2z-1` 下有 5 个系数 `q=[a0,a1…a4]`。对 5 个 coefficient shares 加 1 个 slack share 做 softmax；`q_i=B·share_i·tanh(raw_i)`，`B=0.5-margin`。因此 `Σ|q_i|<B`。
- Normal XY 的 5 个二维 coefficient vectors 共用 5+1 softmax shares；`v_i=B_n·share_i·r_i/sqrt(||r_i||²+δ²)`，`B_n=1-margin`。因此 `Σ||v_i||₂<B_n`。
- softmax slack、tanh 与 smooth vector normalization 只存在于离线训练。导出按 `u=2z-1` 解析折叠为普通 `Wz+b`；运行时仍只有 35 参数 affine。
- 已知安全系数初始化采用显式 interior inverse：按目标系数绝对值/向量范数分配 share，并把剩余预算的一半均分为正的 share floor、另一半给 slack，再分别使用 `atanh` 或 smooth-direction inverse；禁止靠 optimizer 随机逼近初始化。

### 8.3 M2 P0 测试顺序与安全校准

P0 定义为有效 UV 纹素均匀、mean-centered、**不做逐通道方差标准化**的普通 7D→4D PCA。这样 “PCA” 不会暗中变成 channel-weighted PCA；目标通道顺序与 Learned Linear 完全相同。

| ID | RED 所描述的公共行为 | GREEN 验收 |
|---|---|---|
| P01 | invalid atlas 值无论填什么都不影响 PCA | 与显式抽取 valid rows 的结果一致 |
| P02 | 固定输入产生确定的 mean/components/scores | eigenvector 以最大绝对分量为正的规则消除符号歧义；hash 稳定 |
| P03 | 常量、rank<4、单 valid texel 都能生成有限 P0 | 缺失维度为零 score/component，不抛随机 SVD 结果 |
| P04 | score 使用 valid exact min/max 仿射映射到 `[0,1]^4`，零 span 通道固定 `0.5` | encode/decode 解析关系和量化误差分别记录 |
| P05 | raw PCA 与 deploy-safe P0 是两个明确 artifact | raw 仅诊断；safe P0 才可进 UE/learned 初始化，二者误差增量必须报告 |
| P06 | P0 RGBA8、decoder、manifest 可重复导出/重载 | 输入、mask、chart、raw-PCA、safe-calibration、PNG、decoder 均有 SHA-256 |

raw PCA affine 转为 deploy-safe P0 时，不使用 threshold/if-violation projection。对 scalar coefficient budget 的 `S=Σ|q_i|` 和 Normal group budget的 `S=Σ||v_i||₂`，统一使用连续径向压缩 `q_safe=q·[B·tanh(S/B)/S]`（`S=0` 时因连续极限取 1）；Normal 同式使用 `B_n`。该压缩始终执行、无随机性、无 optimizer 状态，并把 raw→safe 质量差单独报告。随后用 M1 的 interior inverse 精确初始化训练参数。

### 8.4 M3 TV / cube 测试顺序

TV 固定使用 fake-quantize/dequantize 后的 latent、水平/垂直边和 `epsilon=1/255` 的 Charbonnier penalty：

| ID | 行为 |
|---|---|
| T01 | 常量 latent 的 TV 为 0，且梯度 finite |
| T02 | invalid endpoint 或不同 chart 的边完全不计数、不传梯度 |
| T03 | 双岛合成图不会跨 seam 平滑；有效边归一使复制 atlas 后数值不变 |
| T04 | hard UNORM8 forward 与 fake-quant forward 相同，STE backward 非零且 finite |

Cube 固定六个 face-local 右手坐标系 `(T,B,N)`；每面 UV 增量分别沿 T/B，camera 位于 `+N` 看向原点且 camera-up 为 B。训练空间保持 glTF tangent `+Y`，不提前执行 UE Y flip：

| ID | 行为 |
|---|---|
| C01 | 六面各完整覆盖 `[0,1]²`，方向色块能捕获 rotate/flip |
| C02 | 每面 `cross(T,B)=N`，方向法线得到预期 world normal |
| C03 | valid mask 与材质使用同一 UV/sample footprint；invalid screen pixel 对 loss/gradient 为零 |
| C04 | 六面 identical error 按 valid pixel 总数归一，不因 face resolution/空白比例改变 |
| C05 | cube RNG 的前进不改变核心 helmet camera/light/sampler RNG 序列 |

TV/cube 权重不在本对话预先拍脑袋冻结。实现统一校准器：在 P0 固定的 8 组 calibration batches 上，仅对 latent 求梯度，使用各组 gradient-norm ratio 的中位数，令 `lambda = r·median(||g_base||/(||g_ext||+eps))`。分别输出 `r∈{0.02,0.05,0.10}` 的无 optimizer-step 报告；下一次人工确认选定 r 后写入正式 config，同一 lineage 内永不自适应。

### 8.5 M4 resume / 配对测试顺序

| ID | 行为 |
|---|---|
| R01 | 从同一冻结 P0-safe artifact/hash 深拷贝三个独立 child；初值与核心首批样本相同，但 mutable tensor/optimizer/RNG 不共享；P0 不作为第四个 trainable state |
| R02 | 启用 cube 只前进 cube RNG，不改变核心 RNG；L0/L1 无 cube RNG 消耗 |
| R03 | 连续 N+1 steps 与 N steps checkpoint→reload→1 step 的 batch、loss、参数和 optimizer 状态一致 |
| R04 | checkpoint 保存 raw safe params、latent、optimizers、phase、全部 RNG、config/input hash 与 best metadata |
| R05 | parent/config/input/hash 任一不匹配时拒绝 resume，不新建“近似续训” |
| R06 | endpoint 为不可变 step-qualified 目录；rolling checkpoint 与 40k/80k/120k artifact 分离 |

### 8.6 M5 短预检执行顺序与停止点

1. 全部 CPU synthetic tests；既有 DTF/ARC/UE exporter 定向回归。
2. SciFiHelmet 真实 Core-4 的只读 load/P0/export CPU 检查；不读取 formal holdout。
3. CUDA 10-step correctness：L0/L1/L2 各用独立 preflight 目录串行运行；验证 finite、证书、hash、resume 与显存释放。
4. 固定 timing：每候选 20 warmup + 100 measured steps，记录 median/P95 step time、samples/s、peak allocated/reserved VRAM、各 loss 占比；L2 单列 cube 开销。
5. 生成 P0 raw-vs-safe 报告与 TV/cube 三档 gradient-ratio calibration report。
6. 停止并向用户报告，不启动 40k。只有用户确认校准比例、preflight 成本和 P0 安全损失后才创建正式 M6 配置。

短预检任何 failure 均保留独立失败目录和最小复现，转入 diagnose；不得通过删除失败证据、重置 optimizer、放松证书或改用 formal holdout 来“修复”。

## 9. Enhanced Phase 2：部署同构恢复与 L2 侧线诊断

2026-08-07 的 UE 人工检查确认 `P0_ENH_CHROMA4/8` 已部分恢复黄色、蓝色描边及黑/银区域分区；它们明显优于 standard/repair P0，但 generic high-chroma contrast retention 仍只有 `15.73%/16.62%`，不能宣称 global-q4 已全面修复。旧 P0/L0/L1/L2 及其 checkpoint、artifact、hash 与报告继续保持不可变。

### 9.1 研究边界

- 部署主线仍固定为一张 RGBA8、一个 global safe `4→7` affine、35 parameters、140 B、28 MAC、1 resource、1 filtered sample。
- CPCA K4-q4 与 global q6 只作 representation oracle。cluster ID、多套 affine、额外 latent channel 或非线性 decoder 均不进入本阶段部署实现。
- 仅使用固定权重的二次 weighted low-rank factorization仍是在选择一个 rank-4 affine subspace，不能冒充新的表达能力。Phase 2 的额外价值必须来自 fake-quantized box latent、safe-by-construction decoder、非二次 semantic/render objective、分层采样和显式多目标门禁。
- formal holdout 继续封存；所有 audit、调参与选择仅使用 train-only Core-4 和独立 train-only diagnostic split。

### 9.2 E2-A：误差与训练分布审计（无 optimizer update）

| ID | 行为 | 输出 |
|---|---|---|
| E2A01 | 对每个 enhanced candidate 固定拆分 `source→raw-float→RGBA8-quantized→certified-safe` | overall/channel/region MAE、generic contrast、量化增量、安全增量与 hash |
| E2A02 | 输出 CPCA K=1/2/4/8 的 atlas assignment visualization 和 cluster composition；不人工命名黄色 cluster | 判断 piecewise structure 是否对应稳定材质统计，而非偶然色块 |
| E2A03 | 统计当前三相机对 valid texel/chart 的可见次数、视角/法线方向和亮度分布 | never/rarely-visible texel、region coverage、三相机偏置 |
| E2A04 | 统计 canonical cube 每 face/light 的 valid count、reference luminance、direct-lit/near-dark 比例 | 识别背光/低能量样本是否支配 cube L1 |

### 9.3 E2-B：部署同构 constrained factorization

从 `P0_ENH_CHROMA4/8` 各自深拷贝独立 child，直接联合优化 RGBA8 fake-quantized latent 与 safe affine；不得覆盖 P0 artifact。训练使用 generic continuous rarity/chroma weights、opponent BaseColor metric、material group balance 与原 helmet render objective，不加入黄色专用 mask/loss。

严格 TDD 顺序：

1. `F01`：零步 child 与 parent 解码、certificate、hash lineage 精确一致。
2. `F02`：semantic material-only factorization；证明 box/safety/quantization 全程有效，并固定 raw/quant/safe 三段报告。
3. `F03`：在 F02 上加入 helmet render；分别记录 latent 与 affine 两个参数组的每项 gradient norm/cosine，禁止只看总 loss 标量。
4. `F04`：只对 F02/F03 前两名做更强 SO(4) frame search；目标同时包含 quantization 与 safe cost，且 raw reconstruction 必须不变。

每个候选先做 10-step correctness，再做 `1k`；只有 region/chroma、BaseColor、normal、roughness、metallic、HDR/SSIM 和 RGB full-cube range 均无灾难性回退才允许到 `5k`。5k 后先进入 UE 小集合预览；不得自动运行 40k。

### 9.4 E2-C：训练数据覆盖侧线

当前 helmet render training set 只有三个近前方 `64×64` 相机，而 cube 把整张 atlas 重复到六个平面并使用随机灯；两者分布跨度过大。先构造固定、可复现的 train-only camera/light coverage audit，再冻结一个 balanced set：保留原三相机作 anchor，新增环向和高低俯仰视角，并按可见 texel/chart 覆盖报告，不按 UE 截图临时增删相机。

先只运行 `L0-data+` 短候选以隔离数据集效应；若它不能在保持 global material/chroma 的同时改善视角覆盖，则禁止把 data+ 带入 L2。历史 camera31/DTF 结果只作“覆盖会重新分配 C4 容量”的警示，不复用其训练或 formal holdout 结果。

### 9.5 E2-D：L2 近黑故障与修订门禁

旧 L2@40k 已确定性复现近黑：RGB full-cube upper bounds 从 step 1k 的约 `0.222/0.217/0.213` 收缩到 step 10k 的 `0.0367/0.0357/0.0346`，最终为 `0.0311/0.0301/0.0290`。相对 L0，normal cosine `0.0300→0.00485`，但 BaseColor L1 `0.00381→0.05093`、mean HDR MAE `0.000722→0.003894`；这是多目标容量转移和优化故障，不是 UE 导入失败。

修订前按 diagnose 执行：

1. `D01`：在共同 parent 和固定 8 batches 上，分别测 base/cube 对 latent、scalar-affine、normal-affine 的 gradient norm 与 pairwise cosine；旧 `r=0.05` 只按 latent 标定，不足以授权双 optimizer 训练。
2. `D02`：验证 cube-off 与 paired L0 短轨迹一致；再以 cube→latent-only、cube→affine-only 作为诊断，不作为 winner。
3. `D03`：若存在参数组梯度主导，改用冻结的 per-parameter-group scale；若存在负 cosine，再比较延迟/ramp cube 与确定性 conflict projection。不得直接引入未测量的动态权重算法。
4. `D04`：若 E2A04 证实低能量偏置，使用每 face/亮度 strata 平衡及独立 diagnostic light seeds；不得简单删除所有暗部。
5. `D05`：只有 `L0-data+` 通过后，才从其同一 enhanced parent 派生 revised L2；先 1k/5k 和 UE 预览，不续旧 L2 checkpoint。

硬停止信号：RGB range 连续三个观察点收缩、BaseColor/region contrast 明显恶化而 normal/cube 单项改善、或任一参数组 weighted cube gradient 超出冻结目标比例。触发后保存失败输出并停止，不用延长步数掩盖。

### 9.6 Phase 2 执行顺序

`E2-A audit → enhanced parent 人工冻结 → F02/F03 1k → L0-data+ 1k/5k → D01–D04 → revised L2 1k/5k → UE preview → 用户决定是否共同 40k`。

本节只授权方案与可复现审计定义；在新的用户授权前，不创建正式 40k 配置，不启动长训练。

### 9.7 P0 cluster/IRLS 审计结果与停止门禁

2026-08-07 的 bounded Core-4 CPU audit 新增了两类仍符合 P0 零更新语义的 train-only weighting：K=4 material-cluster inverse-frequency weighting，以及连续 residual-tail IRLS。最终仍只部署一张 RGBA8 和一个 global safe `4→7` affine；cluster assignment 不进入 artifact 或运行时。

结果没有产生比 chroma8 更好的 certified 候选。cluster b=.5/1.0 与 residual3 的 safe generic chroma contrast retention 为 `15.68% / 11.04% / 16.52%`，低于 chroma8 的 `16.62%`。residual7 raw seven-MAE/tail BaseColor MAE 改善到 `0.029957/0.006108`，但 full-cube safe 后 retention 降至 `9.57%`；切换 safe calibration 权重仍只有 `9.76%–9.57%`，说明失败来自 raw 子空间的 cube safety geometry，而不是 safe objective 的简单权重选择。

因此当前 certified global-q4 最优保持 chroma8，artifact hash `d9e630cf…b748`、certificate margin `0.001001894`，raw→safe seven-MAE `0.0405096→0.0688511`。它仍只保留 source generic chroma contrast 的 `16.62%`，不能自动冻结为 L0/L1/L2 新 parent。报告位于 `outputs/scifihelmet_c4_affine_v1/pca_audit/a874ad-residual-r3/`，SHA-256 `5c1e6e21…a488`。

`E2-B` 及后续训练继续暂停，等待用户选择接受 chroma8 的部分恢复，或授权改变共同训练基础的模型类别/部署契约。`E2-D` L2 修订明确保持 TODO/deferred，不得在 P0 未裁断时实施或训练。

### 9.8 Camera-relative lighting 修订与短程门禁

2026-08-07 的 exact post-80k sampler audit 证明，camera31 本身已接入训练，但六个点光源仍固定在世界坐标，并与环绕相机独立抽样。12 个连续 RNG draw 中有 6 个光源位于相机对面半球；完整 `31×6` 解析矩阵中有 `91/186`（48.92%）个 opposite-side 组合。camera-side/opposite-side 的 mean linear luminance 为 `0.04011/0.01212`，dark fraction 为 `65.28%/96.34%`，mean positive `N·L` 为 `0.41511/0.02447`。因此下一步不再增加相机数量，而先修正灯光相对坐标和采样分布。

本修订作为独立 `L0-lightrel` lineage，必须保持 chroma8 parent、31 个相机、256²、Core-4、fake UNORM8、safe affine、material+helmet objective、optimizer、LR、batch size、checkpoint 与核心 RNG 不变。只允许改变以下一项：灯光由 world-fixed position 改为以每个 camera target 为原点、以 view-right/view-up/target-to-eye 构造的 camera-relative frame；focus 相机使用自己的偏移 target，不再继续围绕世界原点布光。

执行顺序：

1. **LR01 / TDD 坐标契约**：合成相机证明 canonical front/right/up 在 yaw、elevation 和 focus target 变化后仍映射到预期相机半球；正交性、handedness、距离、颜色、强度和 exact-resume draw 全部锁定。
2. **LR02 / 零更新全矩阵审计**：对 31 个相机和 6 个 relative light family 导出 reference contact sheets、camera-light angle、masked luminance、dark fraction、positive `N·L` 与 per-camera/per-light strata。不得读取 formal holdout，不加载 learned decoder。
3. **LR03 / 冻结采样配额**：六个 family 中至少 4 个为 camera-side key/fill、最多 1 个为 deliberate opposite-side rim，剩余为 side light；按显式 family 配额抽样，不能再由世界位置与相机碰巧决定正反面。deliberate rim 必须有独立最低亮度/ambient 审计，不能产生近全黑训练帧。
4. **LR04 / 分布 gate**：加权 opposite-side 概率不超过 `1/6`；非 deliberate-rim 样本不得落入 opposite-side；所有非 rim family 的 mean dark fraction 不高于旧 camera-side audit 的 `0.70`，且 mean positive `N·L` 不低于 `0.30`。任一 gate 失败只调整零更新 light spec，不启动优化。
5. **LR05 / 1k 配对 A/B**：从同一 certified chroma8 `d9e630cf…b748` 深拷贝一个 fresh L0 child，只使用 revised lighting 跑 1k；与不可变的旧 L0@1k 比较相同 camera/material RNG、generic chroma、固定黄色诊断、BaseColor、七通道、HDR/SSIM、RGB range、证书和参数趋势。
6. **LR06 / 5k 条件续跑**：只有 1k 时 generic chroma retention 不低于 parent 的 90%、黄色 R-B 不低于 parent 的 80%，且 seven-channel/HDR/certificate 无灾难回退，才 exact-resume 到 5k。5k 后输出 parent、旧 1k/5k、新 1k/5k 的隔离 UE 预览并停止。

`L0-lightrel` 首轮不同时加入 chroma guard、TV、cube、PCGrad 或新 PCA，以便单独判断灯光分布的因果效果。如果 revised lighting 仍发生颜色坍缩，下一独立 lineage 才加入通用 BaseColor/opponent-chroma guard；如果颜色得到保护，再以相同 lighting contract 讨论 L1/L2。未经 1k/5k 和 UE 人工裁断，不创建或启动 40k/80k。
