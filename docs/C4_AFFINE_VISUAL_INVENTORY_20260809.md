# SciFiHelmet C4 单仿射进展图素材清点

日期：2026-08-09

## 1. 清点边界

- 只读取 train-only、已公开给当前任务的 source/parent/checkpoint/report 与确定性诊断图。
- formal holdout、sealed/local-only、历史隔离输出不读取。
- 本次只清点，不生成新渲染、不启动训练或 UE。
- 后续综合图中的所有方法必须在同一 camera、light、256²、tone/exposure 和 renderer 下重渲染；旧图只在合同完全一致时直接拼接。

## 2. 已可直接复用的成图

### A. 通用颜色 r=.10/.25 对比

目录：`outputs/scifihelmet_c4_affine_v1/color_guard/bf215f-dual-r010-r025-1k-r1/diagnostics-r1/`

| 文件 | 尺寸 | 内容 | 可用性 |
|---|---:|---|---|
| `render_focus_comparison.png` | 1982×1064 | Source、Parent、C0、C1/C2@.10/.25；四个黄色管线 focus camera，使用各自适合的 side/rear light | 可直接作为“第一轮颜色 guard”证据；不覆盖正视/顶视 |
| `basecolor_atlas_comparison.png` | 1982×290 | 七列完整 BaseColor atlas | 可直接复用 |
| `yellow_basecolor_comparison.png` | 1982×290 | source-defined yellow mask 下的 BaseColor | 可直接复用，但必须标 `selection_metric=false` |
| `yellow_r_minus_b_comparison.png` | 1982×290 | 固定 0–0.25 量程的黄色 R-B | 可直接复用，但必须标 `selection_metric=false` |

四个现成 focus case：

| camera | 定义 | light | 画面用途 |
|---:|---|---|---|
| 19 | `focus_ep20_y120` | `side_right` | 后侧左向局部，黄色双管清晰 |
| 20 | `focus_ep20_y150` | `rear_top` | 最接近建议的“黄色管线后视”，两根管线完整且照明均衡 |
| 21 | `focus_ep20_y210` | `rear_top` | 对称的另一侧后视，可作补充 |
| 22 | `focus_ep20_y240` | `side_left` | 后侧右向局部 |

### B. YC-tail / hue8 风险对比

目录：`outputs/scifihelmet_c4_affine_v1/color_risk/bf215f-tail-hue-r010-1k-r2/diagnostics-r2/`

| 文件 | 尺寸 | 内容 | 可用性 |
|---|---:|---|---|
| `focus_render_comparison.png` | 1724×1064 | Source、Parent、G0–G3；相同四个 focus camera，但统一 light0，整体较暗 | 数值合同严格一致；展示时优先重新使用 r1 的 rear-top 光照重渲染，以提高可读性 |
| `basecolor_atlas_comparison.png` | 1724×290 | Source、Parent、G0–G3 BaseColor | 可直接复用 |
| `yc_tail_error_atlas_comparison.png` | 1724×290 | 各 YC bin 内 worst 25% opponent error，冻结统一量程 | 强烈建议直接复用，是“尾部之间误差转移”的主证据 |
| `hue_group_error_atlas_comparison.png` | 1724×290 | neutral+hue8 group mean opponent error，冻结统一量程 | 强烈建议直接复用，是 G2 改善 hue、恶化其他区域的主证据 |
| `hue_group_membership.png` | 256×256 | source-defined hue group 成员图 | 可作图例/附图，不适合独立承担结论 |

上述 render 与训练报告的 HDR/SSIM/foreground count 已逐值核对，`render_report_exact_match=true`；formal holdout 未访问。

## 3. 可确定性重渲染的历史节点

以下节点都有 source material、RGBA8 latent 与 decoder artifact，或有完整 checkpoint，可在当前 `RGBA8→filtered sample→single affine→GGX` 管线下对统一四视角重渲染：

| 节点 | 资产位置 | 叙事作用 | 建议纳入主图 |
|---|---|---|---|
| Source | Core-4 source + glTF | 完整材质参考 | 必选 |
| Standard P0-safe | `preflight/a874ad-r2/p0-safe/` | 最初 PCA+certificate 基线，展示安全约束代价 | 必选 |
| Enhanced chroma8 parent | `pca_audit/a874ad-enhanced-r1/candidates/chroma8/` | 当前 certified global-q4 最优初始化，chroma retention 16.62% | 必选 |
| L0@40k / L1@40k / L2@40k | `train40k/a874ad-r005/` | 基础、TV、cube loss 的第一轮长跑 | 主图只建议 L0；L1/L2 放附图或省略 |
| Chroma8 L0@80k | `train80k/a874ad-chroma8-l0-camera31-light6-r1/` | global material/HDR 改善但颜色坍缩的最清晰长跑证据 | 必选 |
| Lightrel L0@40k | `train40k/a874ad-chroma8-l0-lightrel-40k-r1/` | 修正灯光后颜色仍坍缩，排除采光主因 | 建议附图，不占主图列 |
| C0@1k | `color_guard/.../runs/C0/.../step-001000/checkpoint.pt` | 同步数、无颜色项 control | 必选 |
| C1-r010@1k / G0-mean@1k | 两者 exact replay | 均值 opponent 的代表节点 | 二选一；统一标为 `Mean opponent, r=.10` |
| G1 YC-CVaR25@1k | `color_risk/.../G1-yc-cvar25/...` | 直接 tail 风险优化但改善不足 | 可放方法附图 |
| G2 hue8-macro@1k | `color_risk/.../G2-hue8-macro/...` | 黄色/worst-hue 恢复最明显，同时 macro YC-tail 回退 | 必选 |
| G3 CVaR+hue8@1k | `color_risk/.../G3-cvar25-hue8/...` | 两种 tail 目标的中间折中 | 建议附图或尾部图 |

## 4. 四个目标视角的现状

| 目标视角 | 现有相机/图片 | 当前结论 |
|---|---|---|
| 正视图 | camera 0：`train_e0_y000`，+Z-front；另有早期 `outputs/reference/scifihelmet/front/reference.png` | camera 0 可用当前统一管线重渲染；早期 500² reference 只作构图参考，不能与新图混拼 |
| 顶视图 | camera pool 最高只有 `ep35`，没有严格顶视 | 唯一缺口；需新增 visualization-only top camera，建议 elevation 75–80°，或 90°并改 up vector；不得进入训练、gate 或 holdout |
| 斜上侧视图 | camera 9：`train_ep20_y090`；也可使用 camera 29：`orbit_ep30_y210` | 建议 camera 9 作为标准上侧视，仍属于既有 train camera |
| 黄色管线后视图 | camera 20：`focus_ep20_y150` + light2 `rear_top` | 已有成熟构图，建议直接冻结；camera 21 可作镜像备选 |

推荐每行使用适配该方向但跨方法固定的光：

- 正视：light0 `front_right`；
- 顶视：light2 `rear_top`，必要时增加仅用于展示的固定 key/fill，但不能改变材质比较顺序；
- 斜上侧：light4 `side_right` 或 light0，先用 source 预览裁定一次；
- 黄色后视：light2 `rear_top`。

## 5. 建议的主图节点

为了既反映历史进展又避免列数过多，首版建议固定六列：

1. `Source`
2. `Standard P0-safe`
3. `Enhanced chroma8 parent`
4. `Chroma8 L0@80k`
5. `C0@1k`
6. `G2 hue8-macro@1k`

这六列分别对应：完整参考、初始单仿射、安全/颜色修订、长跑颜色坍缩、同步 control、目前最明显的 hue-tail 修复。Lightrel、C1/G0、G1、G3、L1、L2 作为第二张“方法消融/失败诊断”图，不挤入主视觉时间线。

建议最终形成三张图：

- **图 A：四视角进展矩阵**：4 行视角 × 6 列节点，全部重新确定性渲染；
- **图 B：颜色目标消融**：Source/Parent/C0/G0/G1/G2/G3 的 BaseColor、黄色 mask 与 R-B；
- **图 C：尾部误差分布**：直接复用 YC-tail 与 hue-group atlas，统一色标，并在 G2/G3 下标注“改善一类尾部、转差另一类尾部”。

## 6. 暂不拿来混图的素材

- 早期 `outputs/reference/scifihelmet/{front,left_30,right_30}`：分辨率与旧 renderer/light 合同不同，只用于选角度。
- UE 编辑器人工截图：可证明实际接入，但光照、后处理和截图布局不适合作为离线算法像素级比较；另做“UE readback/落地验证”附图更合适。
- latent RGBA8 和 readback PNG：适合证明数据一致性，不适合主质量对比。
- L2 cube、失败 repair、未通过的 cluster/IRLS：保留为方法附录，不占主进展图。
