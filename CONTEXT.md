# 可微渲染材质压缩

本语境描述一个面向实时渲染部署的材质表示优化项目，以及围绕该项目组织的学习闭环。

## Language

**参考材质**：由原始 Base Color、切线空间 Normal、Roughness 和 Metallic 贴图定义的质量基准。
_Avoid_: 原图、真值贴图

**潜在贴图**：离线优化得到、供运行时采样并经解码器还原材质参数的低通道纹理表示。
_Avoid_: 压缩图、编码图

**实时解码器**：在材质属性写入 GBuffer 前，将潜在纹理样本映射为 PBR 材质参数的轻量函数。
_Avoid_: 解压网络、编码器

**逐角解码后材质过滤**（decode-then-material-filter, DTF）：对双线性 footprint 的四个量化潜在纹素分别执行共享实时解码器和 Core-4 后处理，再在材质语义域过滤 BaseColor、切线空间 normal、roughness 与 metallic；过滤后的切线法线只归一化一次。
_Avoid_: 先解压整张材质、四张解码贴图

**C4 单仿射解码**（C4 single-affine decode）：对一张 RGBA8 潜在贴图执行一次普通硬件过滤采样，再用每资产独立、无隐藏层的 `4→7` affine 直接恢复 BaseColor RGB、切线 Normal XY、roughness 与 metallic；训练侧安全参数化在导出时折叠为普通 35 参数 `W,b`。
_Avoid_: 线性网络（未说明颜色/过滤/结构）、PCA decoder（PCA 与 learned 共用同一运行结构）

**直接过滤材质通道**（direct-filtered material channel）：以运行时纹理过滤结果直接作为材质参数、不经过实时解码器的通道。
_Avoid_: linear decoder、直通网络

**共享辅助表示**（shared auxiliary representation）：由同一潜在样本和共享解码路径共同恢复多类非颜色材质属性的表示。

**分头辅助表示**（split-head auxiliary representation）：共享同一潜在样本、但以独立解码头恢复不同材质属性组的表示。

**分区辅助表示**（partitioned auxiliary representation）：为不同材质属性组分配互不重叠的潜在通道和解码路径的表示。

**逻辑驻留字节**（logical resident bytes）：按表示实际使用的语义通道数计算的理论纹理字节数。

**实际驻留字节**（actual resident bytes）：由目标图形 API 与引擎实际像素格式决定的 GPU 资源字节数。

**legacy single-sample baseline**：冻结的 pre-QAT hard `4→8→7` 历史比较基线；一次纹理采样、103 参数、412 B 权重、88 MAC 和 16 MiB actual resident。它保留为 comparator，不代表当前推荐部署架构，不得被新 winner 覆盖或删除。

**R0b filter-safe hybrid control**：SciFiHelmet 当前的插值安全对照与画质上界；两次纹理采样，BaseColor/roughness/metallic 为**直接过滤材质通道**，只有 normal 经 `2→6→2` decoder。它证明 direct material channels 可消除主要插值暗斑，但不是已接受的最终贴图通道压缩架构。

**训练渲染器**：提供可微材质采样、PBR 着色和图像损失反向传播的离线渲染实现。

**GBuffer 运行时**：在 GBuffer 材质写入阶段执行实时解码器，并用于画质与性能验证的实时渲染实现；可以是自定义渲染器或 Unreal Engine。

**UE 演示**：为学习现代引擎工作流而在 Unreal Engine 中复现潜在贴图解码的集成产物，不是课程硬性依赖。

**学习证据**：能够证明某项原理已经通过操作、结果和解释得到掌握的可复现记录。
_Avoid_: 学习打卡

**独立复现**：不依赖课程预留 TODO 或 AI 直接生成核心答案，能够说明数据流、从最小入口搭建实现并定位常见错误。
_Avoid_: 跑通示例、完成 TODO

**里程碑闭环**：同一阶段内完成原理理解、工具操作、项目实现、验证和复盘。
_Avoid_: 仅完成代码、仅看教程

## Relationships

- 一个**参考材质**对应一个或多个候选**潜在贴图**与**实时解码器**组合。
- **逐角解码后材质过滤**把非线性解码放在材质过滤之前，结构上避免 `decode(bilinear latent)` 在 footprint 内跨激活区域生成新的材质极值；它以四角 decoder 成本换取质量优先的 filter-safe 数据流。
- **C4 单仿射解码**因 affine 与线性纹理过滤可交换，用一次过滤采样满足导师确认的最终计算预算；标准 PCA 与 Learned Linear 的区别只在离线表示求解，不在运行时 shader 成本。
- **直接过滤材质通道**绕过**实时解码器**；**共享辅助表示**、**分头辅助表示**和**分区辅助表示**描述其余通道的容量共享边界。
- **逻辑驻留字节**用于算法记账，部署选择必须以**实际驻留字节**为准。
- **参考材质**、**legacy single-sample baseline** 与 **R0b filter-safe hybrid control** 分别承担质量标尺、16 MiB/单采样压缩原型和插值安全上界；三者不能通过覆盖资产路径相互替代。后续方案必须同时追求 `actual resident < 32 MiB` 与 filter-safe。
- **训练渲染器**比较参考材质与解码材质在多视角、多光照下的渲染结果。
- **GBuffer 运行时**复现训练阶段确定的采样、解码和材质约定。
- **UE 演示**是 **GBuffer 运行时**的一种可选实现。
- 每个项目阶段产生一个**里程碑闭环**，并保留相应**学习证据**。
- **独立复现**通过“解释—操作—诊断—迁移”四类学习证据验收。

## Example dialogue

> **开发者：**“材质空间损失下降，是否说明这个里程碑完成了？”
> **课题负责人：**“还不够；需要在训练渲染器中验证外观，在部署渲染器中复现解码，并用学习证据解释颜色空间、法线和性能结果。”

**部署同构训练**：训练中的量化、纹理采样、decoder 与 postprocess 操作顺序与部署契约一致；不表示 Python PBR 与 Unreal Engine Default Lit 逐像素相同。

**激活区域一致潜在单元**：局部 `2×2` 量化 latent 单元的四角，对 decoder 的每个 hidden unit 都保持在同一激活区域并留有固定 margin；它是抑制单元内部 ReLU crossing 的局部条件，不等价于系统整体 filter-safe。

## Flagged ambiguities

- “GBuffer Pass 解码”不要求使用 UE；自定义渲染器中的具体验收边界仍需向课程导师确认。
- 参考对话中的网络结构、压缩率和工具版本属于候选方案，不视为课程硬性要求。
- “linear”曾同时指颜色空间、纹理过滤和 decoder 结构；已消歧为 linear RGB、bilinear filtering 与**直接过滤材质通道**三个独立概念。
- “一次纹理采样”只属于 legacy/ARC 极限 ablation，不是课程硬约束；DTF 的四角 point load、纹理资源数、decoder 四次执行和 actual resident 必须分别记账。
- 2026-08-06 导师将最终部署约束明确为 C4 上限、一次无隐藏层 `4→7` affine；因此“一次纹理采样”重新成为新主线约束。上一条只适用于该确认前的 DTF 研究背景，DTF 现在是质量/成本对照而非推荐部署。
- R0b 的 `28 MiB logical raw` 与 UE 的 `32 MiB actual resident` 是两个不同口径；当前不能用前者宣称相对 Core-4 32 MiB raw 比较预算的实际显存压缩。
- fresh 80k 的 `C5-DTF-16` 是当前离线质量主选，`C4-DTF-16` 是更低资源 Pareto 备选；C5 的 HDR/SSIM/material composite 更好，但两资源/8 point loads/20 MiB theoretical raw 且部分伪影与材质指标略退，因此“质量主选”不等于全面 Pareto 支配，也不等于达到 R0b。
- “接触过”不等同于**独立复现**；本项目将 AI 辅助编码与个人掌握程度分别记录。
