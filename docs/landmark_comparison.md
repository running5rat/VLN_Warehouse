# 导航地标感知对照

VLM 每个采样源帧只执行一次物体选择问答，输入当前 RGB 和实际导航任务。
`objects-only` 不执行 Route、waypoint、参考线或上下文压缩。
下游 `yoloe-ppocr` 和 `florence2` 使用相同类别回答，保存独立地图。

## 隔离环境

Isaac Sim 仍通过 `DynaNav/warehouse_loop/run_host.sh` 使用宿主 `/data/isaacsim`。
主进程继续使用 `/data/ticvla/envs/warehouse-io/bin/python`。
新增 `/data/ticvla/envs/landmark-perception`；其本地安装覆盖新模型依赖，读取现有
warehouse 环境的 YOLO/CLIP 包，不修改原环境。PP-OCRv5 的检测和识别采用 CPU、2 线程，
避免 Paddle 与 Isaac/PyTorch 争用 GPU 分配器；计时包含这一选择的代价。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m venv --system-site-packages /data/ticvla/envs/landmark-perception
/data/ticvla/envs/landmark-perception/bin/python -m pip install -r requirements-landmark-perception.txt
printf '%s\n' /data/ticvla/envs/warehouse-io/lib/python3.11/site-packages > /data/ticvla/envs/landmark-perception/lib/python3.11/site-packages/warehouse_dependencies.pth
```

Florence 权重目录默认 `/data/ticvla/models/Florence-2-large`，下载 revision 记录在
`download-revision.json`，模型/代码哈希保存在每个 worker 的 `ready.json`。
本轮 Florence revision 固定为 `21a599d414c4d928c9032694c424fb94458e3594`。
可使用 `huggingface_hub.snapshot_download("microsoft/Florence-2-large", revision=该值,
local_dir="/data/ticvla/models/Florence-2-large")` 复现下载；不要自动追随 main。
PP-OCRv5 权重缓存位置为 `~/.paddlex/official_models`，实际哈希同样保存。
实验目录包含两个运行环境的完整 `pip freeze`。禁止让默认 OCR 版本漂移替换 PP-OCRv5。

## 单帧

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --llm-mode objects-only --base-model-only \
  --perception-backend yoloe-ppocr --output /data/ticvla/outputs/landmark-frame-new
```

后端可替换成 `florence2`。此单帧入口重新执行真实 VLM；严格后端对照应使用以下共享 QA 回放入口。

## 独立场景与对照

`--landmark-variant` 支持 development、original、color、layout、surface、lighting、occlusion、combined。
每个变体使用独立 USD 覆盖层，原始资产与旧采集不变。surface 将文字贴在立体容器表面；
lighting 降低场景灯光并取消牌子自发光；occlusion 包含旋转和遮挡。
普通重复警告语、无文字对象与新编号都在评测目录中，不作为推理输入。

```bash
bash DynaNav/warehouse_loop/run_host.sh --landmark-variant color --capture-hz 1 \
  --output /data/ticvla/outputs/landmark-comparison-new/capture-color
```

对 development 和其余六个变体分别采集后执行：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_landmark_benchmark --root /data/ticvla/outputs/landmark-comparison-new --stage prepare
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_landmark_benchmark --root /data/ticvla/outputs/landmark-comparison-new --stage qa
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_landmark_benchmark --root /data/ticvla/outputs/landmark-comparison-new --stage infer --backend yoloe-ppocr
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_landmark_benchmark --root /data/ticvla/outputs/landmark-comparison-new --stage infer --backend florence2
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_landmark_benchmark --root /data/ticvla/outputs/landmark-comparison-new --stage truth
```

先用 `--split development` 完成开发 QA/后端试跑，然后冻结，再运行 evaluation。
`prepare` 根据里程均匀选择帧（开发 12 帧、每评测变体 24 帧），不按模型成功筛选。
`truth` 是独立评测入口，仅它读取 supervision 和 catalog，输出待审核 `evaluation/annotations.json`
及编号区域画廊。需要依据源图填写 readable（true/false，无法判断保持 null）与审核说明；
未审核数据不能通过选优门槛。`infer --diagnostic` 使用独立人工类别提示，只用于定位失败环节，
不合并到端到端成绩。`evaluate` 输出 `comparison.json`。
诊断可使用 `--diagnostic --keys color-00000000 surface-00000000` 限定相同的诊断帧；
主评测不允许 `--keys` 筛选。诊断结果保存在独立的 `diagnostic-后端名` 目录与地图中。

也可使用分阶段封装，顺序为 `capture` → `development` → 冻结配置 → `evaluation` →
`truth` → 源图可读性审核 → `report`：

```bash
bash DynaNav/warehouse_loop/run_comparison.sh /data/ticvla/outputs/landmark-comparison-new capture
```

将最后一个参数替换为所需阶段。evaluation 默认每个后端 3 个场景 worker，共用同一 GPU；
设置 `COMPARISON_WORKERS=1` 可串行运行。该设置只改变调度，不改变每帧模型参数。
本轮 latency 是 3 worker 实测，不能与独占 GPU 的速度混用。

严格复用单帧已有 QA 的入口（不重新调用 VLM）：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.run_perception_replay \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --response /data/ticvla/outputs/landmark-comparison-20261010/frame200-a/llm/response.json \
  --perception-backend florence2 --output /data/ticvla/outputs/landmark-frame-shared-answer
```

程序核验源 image ID、RGB/depth 哈希及每个事件来源。`llm_calls_this_run=0` 表示复用保存结果，
不把复用计为新的真实前向。原始回答及其一次调用记录完整保留。

模型结果保存后可独立验证重启与证据去重：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.perception.replay \
  --root /data/ticvla/outputs/landmark-comparison-20261010 --backend yoloe-ppocr
```

该命令创建新的验证地图，每帧重启、重复消费并断言默认静态层为空；已存在输出时拒绝覆盖。
其 CPU 投影耗时属于保存结果的独立回放，不代表在线车辆运动期间频率。

## 本次留出结果（2026-10-10）

实验目录：`/data/ticvla/outputs/landmark-comparison-20261010`。
两种后端各完成相同 144 帧；共享 VLM 回答 139 帧解析成功、5 帧失败，失败没有补问。
385 个可见标注目标观测中，133 个为可读唯一编号观测，涉及 35 个“场景＋物理对象”。

| 项目 | YOLO-E＋PP-OCRv5 | Florence-2 |
|---|---:|---:|
| 正确读数且成功三维入图的源观测 | 0/133 | 27/133（20.30%） |
| 标注实例检出（mask IoU ≥0.5） | 7/385 | 86/385 |
| 匹配成功实例平均 mask IoU | 0.938 | 0.890 |
| 跨帧确认的唯一身份 | 0 | 0 |
| 已标注目标中观察到的错误确认／错误合并 | 0／0 | 0／0 |
| 感知 p50 / p95 秒 | 2.59 / 6.95 | 81.98 / 168.88 |

耗时对应每个后端 3 个场景 worker 共用一张 RTX 4090；PP-OCRv5 使用 CPU。
这是本次运行配置的实测，不能当作独占 GPU 的模型速度或实时单 worker 帧率。
少量匹配成功的高 IoU 不能掩盖漏检；两者确认身份都为零，也不能把“没有观察到错误确认”解读为可靠性已解决。

Florence 的正确编号源观测分别为：原始 3/27、换色 7/31、布局 5/27、光照 9/31、表面编号 3/17；
遮挡组无可完整辨认的编号，准确率不可计算。按预定入场条件，选择 Florence 继续做绕环工程测试。
该选择用于候选观测链路，不代表持久唯一身份或永久静态锚点已达标。
其 27 个正例已逐项核对到数据库内的同源文字证据，见 `evaluation/persisted-evidence-audit.json`。

主要失败包括 VLM 漏选、A 对新增编号载体漏检，以及 Florence 的 607 次生成截断、14 次 mask/框不匹配、
5 次无效多边形。文字读数变化会触发保守的整组文字签名冲突；当前实现不自动纠字，也不把部分一致读数强行确认为身份。
两套地图均通过 144 帧逐帧重启重放和去重检查。真实模型与接口已贯通，唯一身份确认能力仍未达到本轮目标。

独立人工提示诊断使用相同的两个源帧与 `sign / storage rack / box / container`，A 在换色帧产生
2 个正确编号三维观测、表面编号帧产生 1 个；B 分别为 1 个和 0 个。这些结果不进入上述留出成绩，
也没有用于重新调整并重跑留出提示。它说明 A 的失败不能简单归结为 OCR 读不出数字，类别提示到
实例定位之间也存在明显瓶颈。诊断同时改变了类别词及类别集合，不能仅归因于单复数。
目标匹配按 mask 与文字绑定评分，类别名称未必正确，例如 A 将一个编号载体检出为 `box`。

固定旧采集第 200 帧也完成真实问答及两后端推理；B 复用 A 的同一原文，产生 5 个入图实例，
不额外调用 VLM。旧帧没有实例真值，不报告 IoU。最终 77 项回归测试通过。

## 实时执行

```bash
bash DynaNav/warehouse_loop/run_live.sh \
  --semantic-backend llm --llm-mode objects-only --base-model-only \
  --perception-backend florence2 --render-hz 5 \
  --output /data/ticvla/outputs/warehouse-loop/landmark-live-new --keep-viewer
```

综合场景增加 `--landmark-variant combined`。新后端与 `--shelf-signs` 冲突，不能用旧红白 OCR
补救新模型失败。网页显示文本、绑定状态、mask 和入图结果；`N/A` 表示模型没有输出置信度。
图像对应处理的源帧，可能滞后于车辆，几何更新不等待语义模型。
`--render-hz` 可调整仿真渲染频率（默认 30，必须整除 60 且不低于 5），物理步进仍为 60 Hz，
RGB-D 与几何采样仍为 5 Hz。本轮实测后采用 `--render-hz 5` 降低共享 GPU 的渲染负载；
该设置没有把物理步长改成 0.2 秒，也没有改变车速或绕环路线。

正式选优后自动跑原始与综合场景：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.perception.validate_live \
  --root /data/ticvla/outputs/landmark-comparison-20261010 --render-hz 5
```

评测未完成时拒绝启动；无合格方案时记录 `not_eligible`，不会绕过门槛启动车辆。
实际绕环检查运动期间成功入图、全部几何帧消费、重启重放和证据去重，并保存实测频率。
已有同名输出时拒绝覆盖；再次实验应使用新的实验目录或保存原输出后更换目的目录。

### 实测绕环结果与未通过项

| 渲染频率 | 原场景几何 Hz／运动中语义批次 | 综合场景几何 Hz／运动中语义批次 |
|---|---:|---:|
| 30 Hz | 4.25／0 | 4.26／0 |
| 10 Hz | 4.99／0 | 5.00／0 |
| 5 Hz | 5.00／1 | 5.00／0 |

每次均完成一圈并最终消费 448/448 帧几何。5 Hz 渲染时，原场景运动中唯一的语义批次耗时
67.66 秒、加入 7 个候选实例，运动期间完成频率 0.0112 Hz；停下并排空队列后共 3 批、16 个实例。
综合场景运动期间没有语义批次完成，排空后共 2 批、17 个实例。两场景保存恢复和证据去重通过，
但“两个场景均在运动期间更新语义”的功能验收失败，语义完成频率 1 Hz 也未达到。
最终结果见 `live-validation-render5.json`；此前各档位和一次 10 Hz 参数校验启动失败日志均保留。

模型与仿真在同一 GPU 上竞争，Florence 的多任务生成是本次主要时延瓶颈。
以上命令适合继续检查候选观测链路，不能视为达到实时唯一身份建图要求的配置。
网页和地图发布接口已实现；自动绕环为无人工交互验证，不把发布频率当作浏览器实际绘制频率。

## 代码入口与交付索引

- `ticvla/warehouse/object_io.py`：包含任务的单次物体问答、列表解析、源请求绑定。
- `ticvla/warehouse/perception/client.py` 与 `worker.py`：隔离模型进程、统一感知返回、两套真实模型任务。
- `ticvla/warehouse/perception/common.py`：裁剪坐标还原、文字与 mask 的归属、歧义和置信检查。
- `ticvla/warehouse/mapping/semantic_frame.py`、`store.py`、`identity.py`：源帧投影、持久事件、跨帧身份证据与冲突。
- `DynaNav/warehouse_loop/live*.py`：几何／语义独立调度、最新帧替换、检测和入图可视化。
- `DynaNav/warehouse_loop/landmark_variants.py` 与 `run_comparison.sh`：独立场景、采集与分阶段实验。
- `ticvla/warehouse/perception/evaluate.py`、`replay.py`、`validate_live.py`：独立评测、重启去重和实测绕环门槛。

实验根目录的 `index.html` 汇总逐帧并排图、地图、三维图、固定帧、诊断和各次绕环报告；
`comparison.json` 保存逐场景指标，`acceptance.json` 区分接口、模型比较、重启验证与实时未通过项。
`implementation-sha256.json`、环境锁文件、各 worker 的 `ready.json` 分别保存代码、依赖与实际模型来源。

## 数据语义与限制

`perceive(observation,categories)` 通过隔离 worker 的 JSON/NPZ 传递，只暴露源 RGB、image ID 和 hash。
YOLO 类别来自 VLM 提示；Florence 以多个任务调用完成定位、区域多边形分割和区域文字识别。
Florence 无置信度的字段为 null；截断结果拒绝，不把外接分割器的输出算作 Florence 成绩。

文字绑定使用源图多边形与实例 mask：覆盖不少于 80%，优先最小且明确嵌套的承载物体；
相似面积或非嵌套候选歧义不绑定。OCR 分数小于 0.8 的读数保留但不作身份证据。
这些是通用几何/置信检查，不使用牌子颜色、布局、编号格式或模拟器身份。

三个不同源帧、一致文字及至少 0.15m 位移或 10° 相机方向差，才标记 identity_confirmed。
同一位置的冲突读数全部保留；不同位置的相同文字标记 duplicate_text，不按编号合并。
文字是属性，不保证其语义为序号；型号、品牌等也可能稳定重复。

所有新后端地标保持 tentative。identity_confirmed 不等于 verified/static，不进入默认静态 MapView。
`MapStore.observed_landmarks(category=...,text=...)` 提供候选查询，位置仍属于各自 submap，跨 epoch
需要已验证的坐标变换。needs_reobservation 仅为后续补看接口，本轮不控制车辆。

几何源自同帧深度，位置为可见表面中心。回投影自洽不等于绝对精度。
统计只对独立标注对象报告 IoU/身份，未标注对象不被默认算作误检或正确识别。

本轮标注范围为新增的牌子与容器，实例真值来自独立仿真标签；文字可读性由助手检查源图裁剪，
审阅决定保存在 `evaluation/visual-review-decisions.json`，不冒充外部人工标注。
普通仓库物品的完整用途标注尚不存在，因此类别覆盖报告为标注目标类别族的近似覆盖，
背景类别比例仅衡量明确的地面/墙面等背景，不能当作所有剩余类别的导航价值准确率。
部分变体同时使用新编号组合，分组差异不能全部归因于单一外观因素。
遮挡组无法完整辨认的字符串不评分；可读分母为零时返回 null，不报告虚假的 0% 或 100%。
`correct_3d` 是读数正确且成功入图的源观测数量；`confirmed_correct_unique_landmarks` 才是满足
跨帧规则并与已标注唯一身份匹配的持久地标数量。两项不能混用。
