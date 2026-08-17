# 简单非金属材质候选清单

日期：2026-08-09

## 目的与筛选条件

按教师建议，先用低结构复杂度、非金属占主导的模型判断 `2048² RGBA8 + single 4→7 affine` 在简单材质上的可达质量，再逐级增加材质复杂度。首轮优先要求：

- 单 mesh、单 primitive、单 material；
- Core glTF 2.0，不依赖材质扩展；
- 有 `TEXCOORD_0`、normal 和完整 BaseColor/roughness/metallic 输入；
- metallic 绝大多数接近 0；
- CC0，来源和下载哈希可复核；
- 几何与贴图规模足以暴露颜色、normal 和 roughness 压缩误差，但不先引入多材质选择问题。

## 已下载候选与分级

| 顺序 | 资产 | 来源/许可 | 结构 | 贴图 | metallic 全图审计 | 决定 |
|---:|---|---|---|---|---|---|
| A0 | CeramicVase01 2K | Poly Haven / CC0 | 1 mesh、1 primitive、1 material；6,590 vertices、10,296 tris | Diffuse、GL Normal、ARM；2K JPG | mean `0.001787`；`99.832% < 0.05` | 首轮最简单纯非金属；BaseColor 变化较弱，适合先验证 roughness/normal 与完整管线 |
| A1 | Avocado | Khronos / CC0 | 1/1/1；406 vertices、682 tris | BaseColor、Normal、RoughnessMetallic；全为 2K PNG | mean `0.000000679`；`99.9994% < 0.05` | 首轮纯非金属；几何最简单，但具有明显绿色/黄色/棕色 BaseColor 变化 |
| A2 | BarramundiFish | Khronos / CC0 | 1/1/1；2,188 vertices、3,864 tris | BaseColor、Normal、ORM；全为 2K PNG | mean `0.000001775`；`99.9982% < 0.05` | 首轮纯非金属；颜色、roughness 和 normal 变化比 Avocado 丰富 |
| B0 | Corset | Khronos / CC0 | 1/1/1；11,505 vertices、18,324 tris | BaseColor、Normal、ORM；2K PNG | mean `0.238805`；`71.395% < 0.05` | 已下载但不进入首轮；作为织物主体加金属配件的混合过渡项 |
| B1 | WaterBottle | Khronos / CC0 | 1/1/1；2,549 vertices、4,510 tris | BaseColor、Normal、ORM、Emissive；2K PNG | mean `0.405761`；`58.697% < 0.05` | 已下载但不进入首轮；作为 metallic/emissive 混合过渡项 |

`metallic` 数值是整张贴图审计，用于快速筛选；正式实验仍要在 rasterized valid UV texels 上重新统计。

## 首轮建议顺序

1. **CeramicVase01**：验证通用 loader、切线重建、Core-4 预处理、P0 raw/safe 和 render round-trip。它的颜色复杂度低，若仍失败，优先检查实现或 normal/roughness 表示，而不是先归因于复杂色相。
2. **Avocado**：在极低几何复杂度下检查非金属多色 BaseColor；用于判断 SciFiHelmet 的失色是否主要来自混合金属材质和全局多语义竞争。
3. **BarramundiFish**：提高颜色、roughness、normal 的空间复杂度，但仍保持单材质和近零 metallic。
4. 三者完成相同的 rank–distortion/P0 审计后，再决定是否进入 B0/B1；当前不启动训练。

建议每个资产先只运行：source audit → valid-UV Core-4 → global q1…q7 raw PCA → full-cube safe gap → 固定视角 reference/P0 对比。只有 P0 证据明确后才建立短训练，不直接复制 SciFiHelmet 的 40k/80k 预算。

## 接入注意事项

- CeramicVase01 node 为 identity，但 glTF 未提供 `TANGENT`；必须用 position/UV 导数确定性重建，不能静默使用任意切线。材质为 `doubleSided=true`。
- Avocado、BarramundiFish、Corset、WaterBottle 均提供 `POSITION/NORMAL/TANGENT/TEXCOORD_0`，但 node 带 quaternion `[0,1,0,0]`；通用 loader 必须显式应用或规范化该变换，不能继承 SciFiHelmet 的 identity-node 前提。
- BarramundiFish BaseColor 是 RGBA，但材质仍为 Core opaque；首轮 Core-4 只读取 RGB，alpha 需在 manifest 中注明未使用。
- BaseColor/Diffuse 按 sRGB 输入并且只解码一次；Normal、ORM/ARM 按线性数据读取。Poly Haven 的 Normal/ARM 虽为 JPG，也不得做 sRGB 解码。
- CeramicVase01 的 ARM metallic 因 JPG 有微小非零噪声；是否阈值为 0 必须作为预处理决定记录，不能在 loader 中暗中修改。
- AO 与 emissive 继续排除在 Core-4 首轮之外，但来源槽位必须记录，不能误接入训练真值。

## 来源与复现

- Khronos 官方集合与模型索引：[glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)、[模型清单](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Models.md)。下载时 `main` 为提交 `2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf`。
- 模型页：[Avocado](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Avocado)、[BarramundiFish](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BarramundiFish)、[Corset](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Corset)、[WaterBottle](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/WaterBottle)。四者 metadata 均声明 SPDX `CC0-1.0`。
- Poly Haven：[Ceramic Vase 01](https://polyhaven.com/a/ceramic_vase_01)、[CC0 许可](https://polyhaven.com/license)、[公开 API](https://polyhaven.com/our-api)。下载采用 API 返回的 2K glTF 及依赖，五个核心文件的 size/MD5 全部与 API 元数据一致。
- 本地入口和逐文件 SHA-256：`assets/source/SIMPLE_NONMETAL_DOWNLOAD_MANIFEST_20260809.md`。

## 当前边界

- 本轮只完成搜索、下载、许可和静态数据审计；没有预处理、渲染、PCA、训练、UE 导入或 formal holdout 访问。
- 下载内容位于 ignored `assets/source/`，不会进入提交；本清单只记录可复现来源和筛选决定。
- SciFiHelmet 历史输出、checkpoint、UE Config dirty 和 sealed/formal-holdout 均未修改。
