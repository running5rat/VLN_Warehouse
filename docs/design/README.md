# v0.7 设计文档构建

[v0.7 PDF](../../dynamic-warehouse-vln-design-v0.7.pdf) 于 **2026-11-23** 按 Model Pipeline、Training Schedule、Data Engine、Deployment Optimization 四部分重组，纳入前视VLM／多方向AE、两点RGB锚点、训练对照、数据配比及轨迹Oracle设计。[Markdown 主源稿](../../dynamic-warehouse-vln-design-v0.7.md) 是后续维护入口；[旧十章源稿](archive/v0.7-20261028.md) 归档，原 [v0.6 PDF](../../dynamic-warehouse-vln-design-v0.6.pdf) 不修改。

## 构建

从仓库根目录运行，使用独立文档环境，不安装到 Isaac 或模型环境：

```bash
python3 -m venv /tmp/vln-design-docs-env
/tmp/vln-design-docs-env/bin/pip install -r docs/design/requirements.txt
/tmp/vln-design-docs-env/bin/python docs/design/build_v07.py
/tmp/vln-design-docs-env/bin/python docs/design/verify_v07.py
```

需要 DejaVu Sans/Mono 字体及中文 TrueType 字体。脚本默认寻找系统或宿主的 `DroidSansFallbackFull.ttf`；其他机器可显式指定：

```bash
/tmp/vln-design-docs-env/bin/python docs/design/build_v07.py \
  --cjk-font /absolute/path/DroidSansFallbackFull.ttf
```

构建生成根目录 PDF、全部编号标题的可点击目录和书签，以及 `v0.7-build.json`。当前正文包含十一幅图，新增 3.3 示教截取、连续任务地图记忆和静态 Oracle 方案图。目录和书签支持四级编号标题（例如 3.3.2.3）。

2026-11-22 的工程细节版保存在 [归档源稿](archive/v0.7-20261122.md)，本版移除代码落点、运行命令、存储字段与低信息密度流程图。报告保留输入图像、源稿、PDF、基线、构建器和字体的哈希，供复现；本次不运行模型或更新训练数据。

## 核验

`verify_v07.py` 检查基线/构建哈希、四部分结构、全部编号标题的可见目录与书签、十一幅图及标题、插图数据哈希、页内文字边界、Unicode替换字符和两点协议，并渲染每一页。默认预览保存在 `/tmp/vln-v07-qa`，可用 `--renders` 指定新目录；自动检查结果写入 `v0.7-verification.json`。这些检查不能代替视觉检查图表、换页和代码块。

内容与状态核对采用仓库现有代码及规范，尤其是 `docs/vlm_annotation_and_io_spec.md`、`Agent.md`、语义地图闭环和训练阶段报告。文中的历史结果均有范围说明；这次不运行模型或训练；仿真仅重新渲染相机视图，不产生新的导航能力验收结论。

## v0.6 原图复用与目录跳转

当前总图直接读取 v0.6 PDF 第 4 页的原始矢量图（原图区域 `[76,269,520,716]`；向右扩展至 580 以放置地图结果），保留模块位置与模式分支；按当前接口删除 Controller 的状态／障碍旁路，仅保留所选轨迹输入。在 Source RGB、Semantic Annotation 与 Reference Decoder 输出旁加入仓库示例，并把 Flow／Scorer 的当前图像输入扩展为多方向感知。图像嵌入参考 TIC-VLA 的视觉表达；未复制其模型结构或能力声明。

可单独查看：[架构图 PNG](figures/model-pipeline-v07.png)、[架构图矢量 PDF](figures/model-pipeline-v07.pdf)。[增强后的 v0.6 原图](figures/v06-enriched-overview.pdf) 为可复现构建中间产物。仓库 RGB 和检测来自已有演示；参考线输出仅保留原 RGB 上的轨迹叠图，是输出形态示意，不是 Decoder 实测输出。Semantic Map 输出连线右侧展示实际保存的最终语义地图；该结果图不充作第 197 帧的因果地图输入。左右图片来自同一车辆记录位姿下相机朝向 ±90° 的真实 Isaac 渲染，不是图片旋转或示意图。

为保持原图可读性，总图按原比例排版；框内原始描述随 v0.6 保留。模块详细算法仍以 v0.7 正文为准，候选检测与 Landmark 语义准入的区别见 1.3.1。

构建器为全部编号标题增加覆盖标题、引导点和页码的整行内部链接，跳转到对应标题位置；目录书签同时保留。验证器检查整行链接数量、目标页、VLM 子节、Scorer、Oracle 章节和原图复用来源。

论文核对来源：

- https://arxiv.org/html/2602.11598v1 （图 2；第 2、3、4、6 节）
- https://arxiv.org/html/2602.02459v2 （图 2；延迟语义接口）
- https://gwxuan.github.io/AwareVLN/ （方法图与关键节点推理）

模型改进分析见 1.5，数据引擎对照见 3.7。Training Schedule 按 Stage1／Stage2／Stage3 展开；文档四部分表示方案结构。

Controller 唯一输入为 Scorer 选出的 traj；原图的 `Current state / obstacles` 输入箭头及 `Safety Shield` 子标签已移除，环境与状态检查归于上游 AE／Scorer。

## 场景图片复现

新增素材位于 `figures/warehouse-assets/`：`left.png`、`right.png`、`front.png`、`multiview-provenance.json`、`final-semantic-map.png` 与 `final-map-provenance.json`，以及 `depth.png` 与 `depth-provenance.json`。相机报告记录场景、源帧、外参和图片哈希；地图报告记录只读数据库来源、观测数与哈希。最终地图展示候选地标，不将其宣称为已验证静态锚点。

在可正常运行 Isaac Sim 5.0 的宿主 Python 环境中，设置仓库为工作目录与 PYTHONPATH 后运行（输出目录必须尚不存在）：

```bash
/data/isaacsim/python.sh docs/design/render_warehouse_views.py \
  --capture /data/ticvla/outputs/navigation/codex-continuous-20261112-v2/capture \
  --input demo/semantic_mapping/warehouse-navigation/frames/00000197/input.json \
  --output /data/ticvla/outputs/document-assets/v07-multiview-new
```

地图仅从现有结果重新绘图，不运行分割、文字识别或地图更新：

```bash
/data/ticvla/envs/warehouse-io/bin/python docs/design/render_final_semantic_map.py \
  --map /data/ticvla/outputs/navigation/codex-continuous-20261112-v2/navigation/map \
  --output docs/design/figures/warehouse-assets
```

此处左右视图共享原相机光心，只改变朝向；说明多方向输入的场景内容，不等于已完成多相机硬件外参标定或导航运行验收。

最新总图直接以深度图替换 Depth camera 文字输入，将 Geometry Fusion 改名 SLAM，并在输出线上展示占据地图。VIO 不展示输入，Semantic Projection 不含 Align 小字，VLM 不含 Target loss 行。深度图源于第 197 帧真实数组；占据地图现使用全 episode 的 1,315 帧已保存几何与源相机位姿，射线记录自由空间、障碍命中优先；与语义图共享全局栅格、坐标轴、裁切范围和显示比例，不宣称新增 SLAM 定位验证。之前的 25 帧局部占据图保留为历史素材，不再用于总图。

总图在输出连线旁原文显示同帧 `parsed.json` 中的 Landmark、Local、Route，并在 VLM 中显示两点 waypoint。Landmark、Local、Route 沿用已有 Codex Oracle 文本；图中 waypoint 按 v0.7 两点方案绘制，坐标为 (480,650)、(480,490)，与 RGB 参考线叠图及锚点图共用常量。原始五点回答文件保持不变，图示调整不冒充新推理；该帧 Decoder 未执行。

占据地图与复现：`figures/warehouse-assets/occupancy.png`、`occupancy.npz`、`occupancy-provenance.json`。输出来源哈希另存于构建报告的 `displayed_output_source`。

```bash
PYTHONPATH="$PWD" /data/ticvla/envs/warehouse-io/bin/python docs/design/render_occupancy_preview.py \
  --capture /data/ticvla/outputs/navigation/codex-continuous-20261112-v2/capture/raw \
  --input demo/semantic_mapping/warehouse-navigation/frames/00000197/input.json \
  --output docs/design/figures/warehouse-assets
```

## 全局地图配对与版式

`render_global_map_pair.py` 从同一个只读地图数据库生成 `global-occupancy.png` 和 `final-semantic-map.png`。两图使用相同栅格、范围与像素尺寸；语义图仅叠加已有地标记录，不添加或修正识别结果。未知、自由、占据三态分别绘制；保存的静态场景中历史障碍命中优先，未将未观测区域补成自由。它们都是最终结果示例，不能作为第 197 帧的因果输入。

```bash
/data/ticvla/envs/warehouse-io/bin/python docs/design/render_global_map_pair.py \
  --map /data/ticvla/outputs/navigation/codex-continuous-20261112-v2/navigation/map \
  --output docs/design/figures/warehouse-assets
```

配对核验见 `global-map-provenance.json` 和 `global-occupancy.npz`。旧 `render_final_semantic_map.py` 为历史表面点云画法；维护当前配图应使用上述配对脚本。版式保留 v0.6 主拓扑，统一 SLAM 与 Semantic Projection 的列中心及框内垂直居中，调整地图卡片尺寸、融合节点留白与折线，沿用 TIC-VLA 参考中的实景缩略图、功能分组和输出示例表达。

## 方案正文的表述约定

正文与图注直接描述模块、表示和数据流，移除素材来源、历史运行状态和“仅为示意”等过程说明。素材溯源及能力状态统一保留在本文件、`v0.7-build.json` 的 `displayed_output_source` 和各素材 provenance 文件中。精简文字不改变素材来源：语义输出框来自第 197 帧 Codex Oracle 历史标注，waypoint 和 Reference 配图按两点方案绘制，该帧 Decoder 未运行；两张全局地图为演示完成后的结果，并非该帧的历史输入。v0.7 训练与部署协议仍以正文规定的两个锚点为准。

## 2026-11-24 图示接口统一

- `FIGURE_WAYPOINTS` 统一总图文本、RGB 参考线锚点和独立锚点图，均为两个点。
- Semantic Annotation 删除图片后的模块框；其检测结果以实线进入 VLM，VLM 的 Landmark 以独立虚线进入 Semantic Projection；mask 通路保留。
- Local 位于 Route 下方，VLM 输出经反馈线分别进入两项状态，再作为下一轮输入。Local 不再混在语义建图的 Landmark 输出框中。
- 构建验证检查总图 waypoint 数量、坐标一致性及源标注文件哈希；源数据与运行代码均不修改。

## 2026-11-24 方案表述与参考线展示

- 图 2 以同帧 RGB 并排展示两个锚点和连续参考线投影，删除车体系坐标网格；内部参考线仍采用车体系度量坐标。
- 四部分正文统一为模块职责、表示、算法和实验设计的描述，去除面向模型或执行人员的命令与建议式措辞；协议字段、参数、监督边界和扩展方案保留。
- 参考线配图沿用已有图示像素曲线，不新增 Decoder 推理或几何精度声明。素材来源及未验证状态仍记录在本文件和构建元数据中。

## 2026-11-24 Hydra Scorer 方案更新

- 删除原 1.5“与相关架构的联系”；后续 Model Pipeline 节号前移，原 1.7 现为 1.6。
- 按用户确认保留九模态 Flow，将 Scorer 改为 Hydra-MDP 式示范偏好与多指标蒸馏；未采用其固定轨迹词表。
- 仓库适配包括 Flow 动态候选、多方向感知、参考线条件和 Reference 分项。总图 Scorer 标为 Hydra heads；Controller 仍仅接收 traj。
- Stage3 和 ScorerOracle 同步补充多头监督、有效标签掩码、学生候选分布与规则／单总分／多指标对照。
- 依据为 [Hydra-MDP v4 的 2.2–2.4 节](https://arxiv.org/html/2406.06978v4)。文档明确区分原论文固定词表与本方案动态候选，以及仓库适配的评分聚合式。
- 此次仅修改方案文档及插图构建，不修改运行模型或声称已完成 Hydra 训练。


## 2026-11-24 四阶段训练图

- 2.1 的参数更新表替换为训练流程图，暖色／火焰表示更新、蓝色／雪花表示冻结、灰色表示未参与。
- Stage1a 仅语言侧 LoRA／全量 SFT；Stage1b 冻结完整 VLM，仅训练 Reference Decoder；Stage2 训练 FMAE、动作条件模块与 Hydra Scorer；Stage3 仓库闭环更新动作／评分，冻结 VLM 和 Decoder。
- Stage2 包含 Flow 预热与动作／评分交替更新，评分梯度在候选轨迹处截断；评分数据由新候选快照刷新，不单列原 Stage3 评分阶段。
- 数据配比和损失同步调整；Stage1b 仅使用几何监督样本，Stage3 使用闭环纠正与历史回放。当前 Stage1 配方及仓库数据安排见下方“统一 Stage1 数据配方”。
- 独立插图：[PNG](figures/training-schedule-v07.png)、[矢量 PDF](figures/training-schedule-v07.pdf)。训练图的模块／冻结图标布局参考 TIC-VLA Figure 3 的多阶段训练面板；核对的 v2 中该面板编号为 (c)，(b) 为价值网络。
- 参考来源：https://arxiv.org/html/2602.02459v2#S3.SS3 。本图为本项目四阶段方案的原生矢量重绘，不复制论文图片，不表示已执行训练。

## 2026-11-24 按训练阶段重组

- 保留 2.1 的训练总览及示意图；2.2 为 Stage1（含 1a／1b），2.3 为 Stage2，2.4 为 Stage3。各节分别组织训练方法、数据配方和消融设计。
- 删除原 2.6 评价标准、2.7 ABot-N0 训练对照、4.5 ABot-N0 部署对照及 4.6 参考依据；保留正文算法来源链接。
- Context Compressor 从模块表移除；K/V 到 z_sem 的处理移入 FMAE 内部条件融合，训练图与部署接口文字同步。此变更为方案组织调整，不删除运行代码中的已有实现。
- 验证器检查训练目录层级、指定章节删除、内部条件融合归属以及 PDF 的完整目录链接。

## 2026-11-24 统一 Stage1 数据配方

- Stage1a 合并为 R2R 72%、英语 RxR 18%、针对性 VQA 10% 的单一配方，不再划分初始、扩展与仓库适应配方；规模消融保持来源比例一致。
- Stage1b 沿用导航数据池，仅采样几何监督合格样本，R2R／英语 RxR 比例为 80%／20%。
- Stage1 不额外建设大规模仓库训练集，已有人工示教保留为迁移检查；Stage3 集中使用仓库闭环数据。
- Stage3 仍冻结 VLM 与 Reference Decoder，因此只直接适应动作与评分。语义或参考线迁移失败进入定向 Stage1 修订，修订数据与迁移检查样本隔离。

## 2026-11-24 独立四阶段与数据来源（历史调整）

- 阶段编号统一为 Stage1 VLM、Stage2 Reference Decoder、Stage3 FMAE＋Hydra Scorer、Stage4 仓库闭环；上方记录使用旧编号。正文、目录、训练图与验证器同步更新。
- Stage1 单一配方保持 72% R2R、18% 英语 RxR、10% 辅助监督；辅助池细分为 5% 回放派生关系／地图问答、3% RefCOCO／RefCOCOg、2% GQA 空间关系。Stage2 复用导航池的有效几何标签，按 80%／20% 采样。
- Stage3 记录 DynaNav／Isaac 仿真、SCAND／GNM 机器人子集、CityWalker 人类视频的来源及监督条件。60／30／10 是审核后的混源目标；首轮完整多方向基线采用合格仿真，缺失传感器或标签的外部数据不计为完整监督。
- Stage4 的 50% 新纠正＋50% 历史回放用于 DAgger 监督更新；RL 使用当前策略交互，二者分开描述。语义错误回流 Stage1，参考线错误回流 Stage2。
- 比较依据为 [ABot-N0 §4](https://arxiv.org/html/2602.11598v1#S4)：其联合 SFT 训练推理与动作并回放约 20% 推理数据，本方案在动作阶段冻结 VLM／Decoder。论文数据规模不是本项目已备齐的训练量。
- 此次更新方案和 PDF，不下载新数据、不运行训练、不修改旧实验标签。来源链接与计划采样比例不代表已有合格库存或实证最优配比。

## 2026-11-24 Model Pipeline 章节精简

- 原 1.7 标题简化为“控制”，控制与转向正文保留。
- 原 1.5 短板与改进顺序移至第一部分末尾，以无编号 PS 呈现，不占用目录节号。
- 正文连续编号：原 1.6 Flow／Scorer 改为 1.5，原 1.7 控制改为 1.6；Scorer 子节和验证器同步。

## 2026-11-26 快慢建图与重建数据引擎

- 新增 1.1.2，解释 SLAM 几何与 VLM 地标选择的异步分工、源帧语义投影及融合地图回传；删除原 3.7 对比章节。
- 第三部分补充 LingBot-Map 类位姿／几何恢复、3DGS 外观、Mesh／SDF／NavMesh 可执行几何的分层方案。重建场景用于静态新任务和 Stage3 多方向动态动作生成，3DGS 不单独充当碰撞或动态动作真值。
- benchmark、真实示教、重建任务采用同一观测／监督／来源接口，区分理想与估计位姿、实测与渲染深度、原始与转换轨迹。完整场景与未来轨迹不进入语义教师观测，所有同源衍生任务继承场景划分。
- 标签表明确 Instruction、Landmark／Local／Route、两点／参考线、动作和评分各自来源与核验职责；教师原始递归链与修订批准链分离。
- 来源：[LingBot-Map 官方仓库](https://github.com/Robbyant/lingbot-map)、[Habitat-GS 官方仓库](https://github.com/zju3dv/habitat-gs)、[NVIDIA NuRec 场景说明](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec)。本次仅将方法写入方案，未安装这些模型或仿真扩展，也未生成新训练数据。
- NuRec 当前资产说明分别列出 Isaac Sim 5.1／6.1 版本；本项目运行环境仍为 5.0，正文引用其表示分工，不声称现有环境已支持该资产或渲染接口。实际集成需要独立验证渲染、碰撞、深度与版本兼容性。

## 2026-11-26 Stage4 联合适配与参考线纠错

- Stage4 改为重建仓库中的联合微调、DAgger 与 RL；VLM、Reference Decoder、FMAE 均可训练，Hydra Scorer 更新独立指标蒸馏。覆盖上方历史记录中 Stage4 冻结语义／路线模块的安排。Stage2／Stage3 的前序冻结策略保持不变，训练图同步显示 Stage4 全部模块更新。
- VLM 按已选 LoRA／SFT 范围更新，不把“模块可训练”混同于无条件全量解冻所有视觉基础权重。离散文本使用策略梯度，随机 Flow 策略的连续条件允许梯度进入 Decoder；联合训练重算当前版本特征，不使用脱离计算图的旧 K/V 缓存。
- Reference Decoder 增加当前前视几何、相对运动和输出年龄条件，以对齐／裁剪处理自车位移，以可信锚点门控和残差控制点学习长度、方向修正。正文明确错误分支和未观测结构的限制，动态避障仍归 FMAE。
- Stage2 来源比例仍为 R2R 80%、英语 RxR 20%；条件采样初值为干净 30%、真实学生预测 40%、合成偏差 30%，延迟另作采样维度。这些是待消融的实验配置，不是已测最优比例。
- 数据引擎补齐源时刻／到达时刻配对、静态 Oracle 恢复路线和干净／可纠错／需刷新样本。所有未来轨迹、完整重建环境仅供监督与独立核验，不进入学生观测。
- 方法依据：[DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[Real-Time Chunking](https://www.pi.website/research/real_time_chunking)、[Flow-GRPO](https://arxiv.org/abs/2505.05470)。Flow-GRPO 原文验证图像生成；导航中的联合优化与静态路线残差纠错属于本项目方案，尚未实现或实验验证。
- 本次仅更新方案源文件、图、PDF 与文档检查，不运行 RL／DAgger、不修改模型与历史标注。

## 2026-11-26 多任务 Stage1／Stage2 数据配方（当前方案）

- 替换此前 Stage1 R2R 72%／RxR 18%／辅助 10%、Stage2 R2R 80%／RxR 20% 配方。Stage4 联合更新设计保持不变。
- Stage1 按有效样本采样：指令跟随 35%（R2R 20、英语 RxR 10、ScaleVLN／SRDF 5）、找物与指代 20%、短程目标 10%、地图记忆／探索／恢复 10%、跟随语义 5%、视觉指代 8%、空间与地图理解 7%、视频时序 5%。总计导航 80%、辅助 20%，比例是待消融配置。
- Stage2 的可靠静态几何子池：指令路线 40%、找物／接近 25%、短程目标 20%、地图／恢复 15%。之前 30% 干净／40% 真实学生预测／30% 合成扰动是条件维度，不与任务桶比例相加。
- 核对来源：[LightNav-0 报告](https://arxiv.org/html/2608.30935v2)、[ABot-N1 报告](https://arxiv.org/html/2607.10383v1)。前者 SFT 导航／辅助为 77.6／22.4，后者覆盖五类导航；没有把原始语料规模当作实际 minibatch 比例，也没有照搬其两个像素目标的语义。
- 新来源包括 HM3D-OVON、REVERIE、ScaleVLN／SRDF、训练场景生成任务、跟随训练回放，以及 PixMo／Ego4D 等辅助数据。公开发布训练标签、可生成的训练场景、评测专用数据分开；TrackVLA 完整训练回放和场景许可仍需逐项核验。RoboSpatial 只引用其生成方法，不把公开 benchmark 当成已开放的完整训练库。
- 导航保留紧凑协议；辅助任务不补造 Local／Route／米制轨迹。静态搜索采用因果观测与探索过程，隐藏目标坐标不进入模型输入。动态追随轨迹不直接充当静态路线 GT。
- 规模按 1,000／3,000／10,000 个导航任务做固定配方对照；不是已拥有数据量或效果承诺。数据下载、付费教师标注和训练均未在本次执行。

## 2026-11-26 按来源说明预处理与标注

- 3.1 来源表的“使用条件”替换为“预处理与标注”，分别写明环境回放／视频采集、位姿与轨迹提取、轨迹 Oracle、VLM Oracle 以及评分标注。
- 将真实环境扫描与重建后的任务生成分开：前者完成多视角采集、度量几何、3DGS 和可通行网格，后者复用环境生成新任务与示范。现有 benchmark 网格直接回放。
- 图像辅助标签和视频时序标签分行说明；已有人工示教保留批准版本，VLM Oracle 仅提供对照。此次不改变训练配方、不执行采集或标注。


## 2026-11-27 Warehouse 章节配图与消融表

- 1.3 展示第 197 帧 RGB／实际检测 mask，以及同一演示 episode 的最终占据与语义地图。当前帧选择 sign；未选中的 storage rack 不随检测自动入语义地图。最终地图是跨帧累计结果，不表示该帧单次生成。
- 两点与连续路线的 RGB 图移至 1.4 Reference Decoder。该图沿用方案示意曲线，不冒充 Decoder 实测预测。
- 1.5 展示真实前／左／右图像和模拟的九模态候选。候选由固定公式绘制，CENTER/FOLLOW 的选择仅作输出接口说明，没有运行 FMAE 或 Scorer。
- 3.1.2 加入“Warehouse demo 场景”的 gsplat 实测留出对比与新视角。素材复制自 `/data/ticvla/outputs/3dgs/warehouse-20261126`，70 个训练视角、11 个留出视角、6,000 步；模拟器位姿和训练帧深度用于初始化。新视角没有图像 GT，留出指标不外推为完整仓库或新布局能力。
- Stage1、Stage3、Stage4 按“消融什么／设计不同参数实验／验证什么”制表。Stage2 仅保留固定配置评价，删除 Reference Decoder 消融。FMAE 仅保留 GT／预测参考线与单候选／九模态；Scorer 单列评分方式对照。
- 配图为 ReportLab 原生矢量布局，生成代码为 `demo_figures.py`，由主构建器调用。`figures/warehouse-assets/demo-provenance.json` 记录 3DGS 素材哈希、FMAE 模拟公式和 Reference 示意状态。
- 独立配图：`mapping-demo-v07`、`anchors-demo-v07`、`fmae-demo-v07`、`gsplat-demo-v07`（均提供 PNG／PDF）。构建和验证命令保持不变。


## 2026-11-27 指定语义图与 R2R-CE 标注实例

- 总图及建图章节的语义结果改为 `demo/semantic_mapping/original-20261013/semantic_bev.png`，完整保留编号、文字和身份状态。建图章节下方用整行展示该图，上方 RGB／mask 仍是候选示例；不将两者描述为同一次帧更新。旧全局占据图保留为几何示例，旧配对资产仅作来源留存。
- 删除原 3.2“多源任务的统一表示”，后续节顺序前移。原 3.3.1 变为 3.2.1“R2R-CE 标准化样本与人工基准”，原 3.3.2 变为 3.2.2“Qwen Plus 教师输入与 Prompt”。
- 以扩至 300 任务批次的 `r2r-10009:00000019` 展示实际标准化产物、USER、图像顺序、SYSTEM 和原始模型回答；模型 ID 为 `qwen3.7-plus`。采用实际请求中的 route-only-preserve Prompt，不使用早期 v6 文档概述替代批次配置。
- `r2r-annotation-example.json` 保存引用原文和来源哈希。例子仅说明归档结果，规则通过不等于人工批准；未运行新 API 调用。历史包中的五个米制 waypoint 与 v0.7 两个像素锚点明确区分，本次不改写训练数据。


## 2026-11-27 标准化处理表与集中输入输出图

- 3.2.1 改为数据来源通用的四列表：处理内容、处理方法（含保存内容）、标准化后的标准格式、标注训练用途。内容覆盖任务分组、时空对齐、关键帧、四时刻 RGB、分割与 OCR、历史地图、未来轨迹参考线、两个像素锚点、递归文本和审核导出。
- 表内张量维度为 v0.7 目标格式，不表示已对全部历史数据执行迁移；旧五米制点归档保留不变。
- 删除 3.2.2 开头配置表；实际 SYSTEM、USER、六张因果图像和原始 JSON 集中到 `annotation-demo-v07.pdf/png`。正文不强调示例数据集；精确来源仍保存在 `r2r-annotation-example.json`，图片副本及哈希位于 `figures/annotation-assets/`。
- 图示完整保留历史 Prompt 和输出，不做新教师调用；图 7 为集中标注示例，部署时序前移后的图号为图 8。


## 2026-11-27 目标部署平台

目标平台统一为 NVIDIA Jetson AGX Orin 64GB，替换封面的工作站＋Orin NX 16GB 组合。4.2 与实施计划同步：车端承载完整推理及控制链路，RTX 4090 保留为训练、仿真和开发设备；频率仍为目标值，不代表已完成 Orin 实测。


## 2026-11-27：3.3 示教轨迹提取与场景任务生成

本节按已有示教与新任务两类来源重组为两个主体小节、五个下级标题。删除原 3.3.4 恢复与延迟纠错小节；模型与训练章节的既有纠错设计不在本次改动范围。三张新图位于 [示教截取](figures/demonstration-demo-v07.png)、[连续任务](figures/memorytask-demo-v07.png)、[静态 Oracle](figures/staticoracle-demo-v07.png)。

`dataengine_figures.py` 只读历史 demo 数据生成素材，文档构建使用已归档的 `figures/dataengine-assets/`，不依赖再次访问 /data。素材重建使用带 NumPy、Matplotlib、Pillow 的分析 Python：

```bash
python docs/design/dataengine_figures.py
/tmp/vln-design-docs-env/bin/python docs/design/build_v07.py
/tmp/vln-design-docs-env/bin/python docs/design/verify_v07.py
```

来源与范围记录在 `figures/dataengine-assets/evidence.json`：

- 示教图使用第 505 帧（转弯前）和第 520 帧（转弯后），后者用同帧深度、标定及实际未来位姿投影。距离上限为既有 32 点、0.25 m 间隔对应的 7.75 m，未修改训练配置。前方约 3.25 m 的近场盲区不绘制像素锚点；真实画面对比展示后续通道进入视野，右下角为独立的遮挡原理示意。
- 连续任务图使用第二任务首个审核帧 951 的真实 RGB 与历史地图；地图截止时间早于查询时间。圈选实例 16（04/5R）只用于展示，不写回模型请求。
- Oracle 图叠加既有 Codex 路线方案与历史实际执行轨迹，用来说明拟议的 Oracle，不代表新运行了 NavMesh／SE(2) 或已完成碰撞验收。俯视背景为完整 episode 的观测几何展示，不作为当前帧因果输入。
- 本次只修改方案文档、构建与核验脚本、配图；原始采集、模型输出和训练标签保持不变。


### 转弯遮挡与 NavMesh 解释补充

图 8 的遮挡局部图改为货架端部右转：可通行路径绕过货架，当前视线在货架角点形成可见边界，边界后段使用虚线。3.3.2.2 分别说明静态碰撞网格到 NavMesh 的离线构建、任务路径搜索、通道中线偏好、真实 footprint／SE(2) 衔接以及示范执行；图 10 模块标签同步更新。该补充为算法方案说明，未运行新的轨迹 Oracle。

### 真实转弯画面对照

图 8 已将直行第 197 帧替换为同一右转片段的第 505、520 帧。真实画面展示视野变化；不得将其描述成已测量的图像内深度遮挡正例。对 mapping 记录每三帧抽检，限定未来路线弧长 7.75 m，未发现多于两个路点满足“位于图像内且路线深度比有效观测深度远 0.30 m”的帧；这不证明场景没有遮挡，只限定本次图示证据。未来路线与模型输入分离，原采集数据未修改。

图 8 第 520 帧的原始 RGB 直接叠加未来轨迹投影（紫色虚线）、深度支持且位于 7.75 m 范围内的参考线（青色实线）、两个锚点及距离截断位置。图像外和相机后方的未来点不绘制，断开的投影片段不跨段连线。

### 转弯原图内标注

图 8 当前使用第 505 帧与第 514 帧。505 帧直接标注右侧视野外的未来路线方向，以及空像素参考线；箭头是文字指示，不是投影轨迹。514 帧仍处于转弯中，按本帧标定和深度绘制真实未来投影及截取参考线，替代此前 520 帧。构建时检查 505 帧所有后续记录位置无正深度图内投影，不在原图上补画不可见轨迹。

### 图 8 单帧版本

按最新要求只保留第 505 帧 RGB，删除其他子图。图内标注实际未来位姿 GT 的画外投影方向及空图内参考线。数据核验显示未来位置不在本帧图像范围内，因此该图说明视野截断，不作为深度遮挡正例；旧版 514 帧投影资产保留供追溯，已不在图 8 展示。

### 图 8 提前至第一次转弯后

最终单图改用第 440 帧，保留真实未来位姿 GT 的转弯形状。深度支持的连续图内投影为弧长约 3.20–6.55 m；6.60 m 处投影为 (961.78, 643.42)，超出宽 960 的图像。图示因此在右边界截断，未触及 7.75 m 距离上限，不标成深度遮挡。紫色虚线底层表示未来 GT 投影，青色实线覆盖其保留段；原始数据不修改。

### 图 8 增加右侧三维对照

左右两图均来自第 440 帧。3D 图以源车体系表示实际未来位姿，显示前 10 m GT（仅图示范围）及同索引截取段，两个锚点与 RGB 一致；灰色点为同帧深度反投影。z 来自实际位姿，不人为抬高轨迹；近场盲区前段及截断后段继续显示为 GT。图示范围不改变参考线的 7.75 m 配置上限。

图 8 右侧按最新要求改为二维俯视图：纵轴 x 向前，横轴 y 向左；GT、截断段与锚点沿用同帧坐标，三维视角不再展示。


### 实测 NavMesh 配图纳入 3.3.2.2

图 10 改用 [真实 NavMesh 多边形](../../demo/navmesh/warehouse-20261127/02-navmesh-polygons.png)，图 11 使用 [跨通道路径查询](../../demo/navmesh/warehouse-20261127/04-path-query.png)，替换旧的静态 Oracle 示意图；部署图顺延为图 12。正文分别说明建网参数、网格颜色、A* 多边形走廊及 Funnel 路径，区分离线静态几何与任务 A 的实际观测地图。此前本节关于图 10 仅为概念示意的记录由本次实测结果更新。

素材副本与来源哈希归档在 `figures/dataengine-assets/navmesh-provenance.json`。实际执行与复现见 [NavMesh 说明](navmesh_demo/README.md)；本次路径查询没有执行车辆控制或 PhysX 碰撞核验。
