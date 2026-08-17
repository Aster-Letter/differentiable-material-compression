# SciFiHelmet Core-4 资产契约

最后更新：2026-08-03

## 范围与真值定义

MVP 使用 Khronos `glTF-Sample-Assets/Models/SciFiHelmet`，许可证为 CC0。源 glTF 包含 1 个 mesh、1 个 primitive、1 个材质和 4 张纹理，并提供 `POSITION`、`NORMAL`、`TANGENT` 与 `TEXCOORD_0`。

课程首版的材质真值只包含 BaseColor、切线空间 normal、roughness 和 metallic，称为 **Core-4 Reference**。源 AO 仅用于检查完整 glTF 外观，不进入首版训练目标，也不能只连接在参考分支上。

## 通道与颜色空间

| 语义 | glTF 来源 | UE 参考材质 | 采样空间 | 首版训练表示 |
|---|---|---|---|---|
| `kd` / BaseColor | `SciFiHelmet_BaseColor.png` RGB | RGB → Base Color | 源纹理 sRGB；着色前转 linear | linear RGB，3 通道 |
| normal | `SciFiHelmet_Normal.png` RGB | RGB → Normal | 非 sRGB；切线空间 | 优先保留 XYZ，后续可用 XY 重构 Z |
| roughness | `SciFiHelmet_MetallicRoughness.png` G | G → Roughness | 非 sRGB、线性数据 | 1 通道，范围 `[0,1]` |
| metallic | `SciFiHelmet_MetallicRoughness.png` B | B → Metallic | 非 sRGB、线性数据 | 1 通道，范围 `[0,1]` |
| AO | `SciFiHelmet_AmbientOcclusion.png` R | Core-4 不连接 | 非 sRGB、线性数据 | 首版忽略 |

BaseColor 的 sRGB 解码只能发生一次；normal、roughness、metallic 不得经过 sRGB 转换。

## UE 5.8 导入与材质设置

- 使用 Interchange `Assets` 管线；通用 GLTF Assets Pipeline 与 GLTF 格式管线按堆栈顺序共同执行，不是二选一。
- 导入静态网格体、材质和纹理；不导入骨骼/动画；关闭 Nanite；不重算法线或切线。
- 开启“检测法线贴图纹理”和导入器默认的“翻转法线贴图绿色通道”，把 glTF 法线约定转换到 UE 使用的约定。
- BaseColor：`sRGB` 开，默认颜色压缩。
- Normal：`sRGB` 关，`Normalmap` 压缩。
- MetallicRoughness：`sRGB` 关，`Masks (no sRGB)` 压缩。
- `M_SciFiHelmet_Reference` 使用 `Surface / Opaque / Default Lit / Two Sided off`；Specular 不连接，采用 UE 默认值 0.5。
- 项目关闭自动曝光，默认曝光模式为 `Manual`、曝光偏差为 0；Post Process Volume 启用并设为无限范围。

## 几何与命名边界

- UE Actor 保持缩放 `(1,1,1)`；导入器负责 glTF 米制到 UE 厘米制及坐标系转换。
- 纹理材质使用 `TEXCOORD_0`；训练端不得另行生成或重排 UV。
- UE 导入资产目标路径：`/Game/Imported/SciFiHelmet/`，可由源 glTF 重建，不进入 Git。
- UE 手写内容目标路径：`/Game/CGCompression/Maps/`、`Materials/`、`Blueprints/`，进入 Git。
- 关卡参考 Actor：`Helmet_Reference`；参考材质：`M_SciFiHelmet_Reference`。

## Python 网格与 GBuffer 边界

- 训练端直接读取源 glTF accessor 与外部 `.bin`，不经 Blender 重导出；保留 glTF 原生右手坐标系、`+Y` 向上、资产正面朝 `+Z`，单位为米。当前 mesh node 及祖先必须为 identity，不能静默忽略 node transform。
- 源 SciFiHelmet 完全按三角形拆点：索引为连续 `0..70073`，每个顶点只被使用一次。POSITION、NORMAL、TANGENT、TEXCOORD_0 与 indices 均按 `bufferView.byteOffset + accessor.byteOffset`、`byteStride`、`componentType` 和 normalized 规则解析。
- 源 TANGENT.xyz 虽为单位长度且 `w` 全部为 `+1`，但与 NORMAL 不构成有效正交基：`max |N·T| ≈ 0.998756`、`mean |N·T| ≈ 0.249957`。Python 渲染不得直接采用该 tangent，也不得把普通 Gram–Schmidt 当成无损修复。
- Stage B 使用 position/UV 三角形导数重建 tangent，再相对源 vertex normal 正交化并计算 handedness；重建结果为 `w=-1` 54,062 个、`w=+1` 16,012 个，`max |N·T| < 4e-6`。源 tangent 仍保留在加载结果和诊断统计中，不被覆盖或伪装成重建结果。
- glTF 纹理坐标 `(0,0)` 对应 PNG 左上角。nvdiffrast 内部图像为 bottom-up，因此 PNG 的 top-down 数组进入纹理采样时保持原行序，使 glTF `v=0` 采样数组第 0 行；raster/GBuffer 离开 nvdiffrast、保存为普通 NumPy/PNG 时再垂直翻转一次。
- TBN 使用 glTF 公式 `B = cross(N, T) * tangent.w`。源 normal PNG 仍不被预归一化或翻 Y；渲染时另保留解码后的 `normal_ts_raw`，并仅为 TBN/光照计算产生单位 normal。
- 固定 Stage-B 配置位于 `configs/render/scifihelmet_gbuffer.yaml`，主分支采用 glTF `+Y` normal；同时输出 Y-flip 世界法线与夹角对照，不让错误约定进入主数据。

## Python Core-4 PBR 参考

- 固定参考配置位于 `configs/render/scifihelmet_reference.yaml`。主图使用单点光、平方反比衰减和固定小环境项；相机、分辨率、光照、曝光与输出清单均由配置冻结。
- BRDF 为 Cook–Torrance GGX metallic-roughness：Trowbridge–Reitz GGX `D`、Smith GGX `G`、Schlick Fresnel `F`，非金属 `F0=0.04`。roughness 仅在 BRDF 数值计算处钳制到不低于 `0.045`，GBuffer 原值不改。
- BaseColor PNG 先逐 texel 执行一次 sRGB→linear，再在 linear 空间双线性采样；normal、roughness、metallic 不执行 sRGB 转换。线性 HDR `.npy` 是参考数值，PNG 仅使用固定曝光、Reinhard 和 linear→sRGB 作为展示。
- 固定错误对照包含 tangent normal Y-flip 与 BaseColor 二次 sRGB 解码。正确分支与错误分支共享网格、相机、光照、BRDF、材质其他输入和显示过程。
- nvdiffrast 路径已验证 BaseColor、normal、roughness、metallic 四张纹理都能从参考图损失获得有限且非零梯度；后续原始/压缩分支必须复用同一 GBuffer/PBR 实现。

## Python legacy RGBA latent 部署与量化契约

- legacy 单采样分支的运行时顺序为：RGBA latent texel 先做 UNORM8 hard/fake quantization，再按 UV 双线性采样 RGBA，随后逐像素 decoder 输出 linear BaseColor RGB、tangent normal XY、roughness、metallic；normal 重建正半球 Z 后经共享 TBN 转 world normal，最后进入共享 GGX。
- legacy/ARC 复现路径禁止先逐 texel decode 七通道材质再采样。tiny MLP 与插值不交换；该限制用于保持旧实验可比性，不约束下面独立的质量优先 DTF renderer。
- UNORM8 使用 half-up：`u8 = floor(clamp(x,0,1)*255+0.5)`、`dequant = u8/255`。fake quant 前向必须逐值等于 hard dequant，反向为 identity STE；量化发生在纹理采样前。
- decoder 的 BaseColor 输出已经是 linear RGB，不再做 sRGB 解码；normal、roughness、metallic 始终为线性数据，normal Y 保持 glTF/OpenGL `+Y`，AO 继续排除。
- 从材质域导出初始化时，`latent_float.npy` 先 clamp 后 inverse-sigmoid 成 logits，训练中使用 `sigmoid(logits)`；decoder 从 `decoder_weights.npz` 加载并允许与 latent 联合微调。
- 原始和压缩分支共享 geometry GBuffer、UV、TBN、camera、light、mask、GGX、minimum roughness 和显示变换，只替换完整材质来源。geometry/material 分离接口与冻结 512² front 锚点最大差低于 `1e-5`。
- 训练/留出配置位于 `configs/train/scifihelmet_render_quant.yaml`。基础 16 个训练相机只覆盖 82.5199% triangles；加入审计选出的 3 个补充视角后覆盖 `20,004/23,358 = 85.6409%`，512² 双线性触及纹素并集为 48.0734%。train/holdout camera-light 参数不得重叠。
- 部署验收必须从真实 RGBA PNG 在新进程逐字节读回，验证 PNG/decoder SHA-256、decoder checkpoint/export 和 hard-render export/reload 等价。RGBA UNORM8 2048² raw GPU bytes 为 16 MiB；Core-4 四张 RGBA UNORM8 理论比较值为 32 MiB。

## Python C4/C5 decode-then-material-filter 契约

- DTF 使用独立 `decode_then_filter_renderer_v1`，不原位修改 legacy 或 `canonical_renderer_v2`。主线从 2048² C4 UNORM8 开始，C5 仅作为条件表示上限；LOD 固定为 0、无 mip、Wrap。
- 存储纹素先执行同一 half-up UNORM8 hard/fake quantization。运行时按 half-texel 规则取得双线性 footprint 的四角，对每个角分别运行共享 `C→W→W→7` decoder，再分别执行 BaseColor sigmoid、normal XY tanh/+Z 重建、roughness sigmoid 与 metallic sigmoid。
- BaseColor 四角结果只在 linear RGB 过滤；roughness 按现有 perceptual roughness 标量语义过滤；metallic 线性过滤。四个正半球 tangent normal XYZ 先过滤，再只归一化一次；训练真值仍为 glTF/OpenGL `+Y`，UE 边界仍只允许一次 `Y×-1`。
- 四角材质进入与参考分支相同的 TBN、minimum roughness、GGX 和 display transform。DTF 不进行第二次 sRGB 解码，也不把 AO 加回 Core-4。
- C4/C5 的 2048² UNORM8 theoretical raw 分别为 16/20 MiB；目标引擎的纹理资源数、point texel loads、实际格式与 actual resident bytes 必须实测，不能从逻辑通道数推断。
- 冻结导出布局为：C4 使用单张 RGBA8；C5 使用 `latent_c5_dtf_16_rgba_unorm8.png` 保存 channels 0–3，并用 `latent_c5_dtf_16_r_unorm8.png` 的 R8 保存 channel 4。C5 因而是 2 个 texture resources、每像素 8 次 point texel load；不得把第二张 R8 静默扩成逻辑 RGBA 后仍按 20 MiB 报告目标运行时 actual resident。
- DTF 使用通用 render/material/quantization/texel-subpixel anchor 目标，不继承 ARC activation-region、commutativity 或固定瑕疵区域专用 loss。双轨 checkpoint 为 `best-render` 与 `best-artifact-safe`。

## 尚待跨实现验证

- Stage B/C 已按规范锁定 glTF `+Y` normal 为 Python 主分支，并用斜向点光生成 Y-flip 图像对照；后续若与 Blender/UE 做视觉复核，只允许修正明确的跨 API 约定，不能反向改写训练真值。
- UE GPU 压缩后的参考图不作为训练输入；训练从源 PNG/处理后的无损贴图读取，UE 用于部署与视觉复核。
- Python PBR 与 UE Default Lit 不要求逐像素完全一致，但同一实现内的原始/压缩分支必须共享相机、光照、几何、后处理和非压缩输入。
