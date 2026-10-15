# 人工样本、Oracle 与训练数据准备

**当前优先级：**人工/API/会话辅助入口见 [README](../README.md#人工与辅助标注第二阶段)。下一步按[静态数据组成与格式对齐计划](warehouse_dataset_alignment_plan.md)执行；下文动态Oracle和训练接口保留为历史实现说明，暂不开展动态标注或正式训练。

**2026-10-15 人工基准更新：** `manual-static-004` 已由用户完成整段批准，并按用户要求建立英文版本。
10 个关键帧的 instruction、Local、Route 描述及输入事件说明已统一为英文，10/10 通过在线生成语法、解析和导出检查。
后文最初实施状态中的“等待首个人工批准”已由该 episode 完成；其他教师、动态 Oracle 和正式训练状态不因此升级。
当前复查仍保留第 1703 帧约 1.9 cm 的 stop/active 参考线、转向关键帧较稀、4 m 到达阈值余量较小等限制。
本 episode 只作人工基准与回归，不同时用于训练和独立评测。

当前语言策略：原始任务、Local、Route.remaining_instruction 和子目标 description 使用英文，
协议键、动作枚举和类别保持英文，场景编号原样保存。界面说明可用中文。
旧采集的中文任务保留在不可变原始记录中；`project.task_translation` 记录原文、英文、来源与授权。
`sample_context` 校验原文和 task ID 后向回放、教师和训练导出提供英文任务，翻译元数据进入 provenance，
不改变原始 source_binding。批准版本链保留翻译前记录与修订来源。

英文复查报告：`/data/ticvla/outputs/dataset-preparation/manual-static-004-english-0fa18e6b/index.html`。
重开原标注工程即可查看英文版本：

```bash
bash DynaNav/warehouse_loop/run_annotation.sh \
  --mode replay --output /data/ticvla/datasets/manual-static-004
```

本轮实现了人工采集/回放工具，以及数据审计、教师调用、官方 ALTRO、九模态 Flow 监督的基础接口。**七步尚未全部验收：首个静态人工任务已批准，schema v2 仍待完整流程验证后定稿。没有启动正式训练，也没有批准的动态 Oracle 专家数据。**

## 1. 先使用人工链路

从仓库根目录执行，不需要先打开 Isaac，也不需要切换到 Isaac Python：

```bash
bash DynaNav/warehouse_loop/run_annotation.sh \
  --mode capture \
  --instruction 'Go straight along the aisle, turn left at its end, then continue straight toward the fire extinguisher you see.' \
  --output "/data/ticvla/datasets/manual-static-$(date +%Y%m%d-%H%M%S)"
```

浏览器打开 **http://localhost:8767**。脚本在当前模型 Python 中启动服务，通过已有宿主桥接运行 `/data/isaacsim` 的 Isaac Sim 5.0。`--headless` 只隐藏 Isaac GUI，浏览器驾驶界面仍可用。输出目录必须是新目录。远程访问可通过 SSH 转发 8767 端口；默认只监听本机，`--bind` 可明确修改。

当前复用原始仓库、Nova Carter、RGB-D 和几何地图。没有动态人物，没有导航 VLM 控制，也没有自动生成语义标签。已有独立语义建图入口未改动。

当前人工基准已确认上述灭火器任务；复现时保持相同起点与场景。更换起点或布局后需重新走查。最初“两次转弯”的示例不是硬性任务模板，不使用隐藏坐标替人判断终点，也不将早期3.38 m自动接口测试当成人工任务。

操作顺序：

1. 等待 RGB-D 和地图刷新，点击“开始／继续录制”。渲染预热期间不允许开始驾驶。
2. 点击“驾驶区域”，按住 **W/↑ 前进、A/← 左转、D/→ 右转**。松开按键或空格停车；失去窗口焦点、传感器过期、命令超过 0.3 秒未刷新也停车。不支持倒车。
3. 在起步、开始转弯、完成转弯、目标可见、停车和确认到达时记录实际事件。每条事件绑定当前图像。完成验收要求目标可见、停车和到达证据，不再硬性要求两个转弯事件；转弯数量由实际任务决定。
4. 可以暂停录制。暂停和恢复之间的轨迹不拼成连续 GT。
5. 点击“结束采集并进入标注”。等待全部已提交帧建图、采集进程退出，页面自动进入回放。
6. 依时间顺序填写 Landmark、Local 和结构化 Route；拖动、插入、删除参考线控制点。填写标注人、勾选审核后批准当前时刻。默认约 10 个关键帧，事件帧会额外加入。
7. 所有关键帧审核后，点击“确认整个任务与标注完成”，再导出全部批准版本。没有充分事件证据的任务不能标为 completed。

人工仿真探索允许进入地图未知区域，由驾驶者判断；仍拦截已知障碍和地图外越界，并记录实际执行/干预原因。这个显式 `human_exploration` 策略不用于自动导航。速度上限 0.3 m/s，角速度上限 0.6 rad/s。

重启回放：

```bash
bash DynaNav/warehouse_loop/run_annotation.sh \
  --mode replay --output /data/ticvla/datasets/manual-static-001
```

也可以先体验**未批准的自动接口夹具**：

```bash
bash DynaNav/warehouse_loop/run_annotation.sh --mode replay \
  --output /data/ticvla/outputs/dataset-preparation/capture-smoke5-20261014
```

回放模式不能驱动车辆。草稿可恢复；修改早期 Route 会使后续依赖的批准版本失效，必须重新审核。所有批准修订独立保存。

参考线编辑区新增“参考线投影到当前 RGB”：橙色参考线和青色五锚点随地图编辑、撤销和恢复候选同步更新。使用当前源帧的 `K` 和 `T_body_camera`，将源车体系 XY 以 `z=0` 平面投影到原图；不使用车辆后续位姿。按实际批准参考线的 32 点/0.25 m 规则采样，统计画面内、画面外和相机后方/近裁剪点。该预览不检查遮挡，不能把图像内投影当成可通行证明；斜坡等非平地场景也不能直接采用此平面近似。

## 2. 目录与格式草案

```text
session/
  session.json                     人工采集配置，未核验任务标记
  capture/raw/                     原始 RGB-D、标定、5 Hz 源位姿
  navigation/executed-commands.jsonl 约 30 Hz 位姿、测量速度、实际控制及原因
  manual-commands.jsonl             人工请求命令、有效期及策略
  recording-events.jsonl            开始、暂停、结束区间
  annotation-events.jsonl           人工事件及源图像
  geometry-performance.jsonl        地图服务与源帧发布延迟
  annotation/project.json          审核进度及 schema 草案版本
  annotation/samples/<frame>/
    source.json                    因果输入和单独的标注辅助
    rgb.png, history-*.png          图像副本
    map.png, map.npz                该时刻的已观测地图
    projection-camera.json         绑定源帧的 RGB 投影标定，仅供预览
    draft.json                     未批准草稿
    approved.json, revisions/      最新批准及不可变修订
  exports/revision-*/               带哈希的导出 manifest 和 v2 样本
```

导出记录的三个命名空间：

| 字段 | 含义 |
|---|---|
| `observation.task` | 固定原始 instruction 和 task ID，不被 Route 覆盖 |
| `observation.rgb_history` | 当前及可用历史图像、源时刻、位姿、图像哈希 |
| `observation.mapview` | 截至源时刻的地图、分辨率、哈希、来源截止时间；当前人工工具只生成几何层，`landmarks=[]` |
| `observation.route / route_revision` | 上一个已批准时刻的 Route 和版本，初始值为原始指令 |
| `observation.execution` | 截至当前源时刻的人工事件、源位姿测量的位移/路程/转角；不把命令当作完成事实 |
| `supervision.text / parsed` | 完整五标签文本和解析结果 |
| `supervision.reference_state` | `unlabeled`、`valid`、`valid_empty` 三态 |
| `supervision.reference_xy / reference_mask` | `[32,2]` 米制参考线和有效前缀 mask；通常间距 0.25 m，末端保留短段 |
| `supervision.waypoint` | 同一批准参考线 20/40/60/80/100% 弧长处的五锚点，或空列表 |
| `supervision.action_state / candidates` | 首个人工样本固定 `unlabeled / []`，不能作为停车标签 |
| `provenance` | episode、布局族、源绑定、标注修订、审核人、数据用途 |

车体系 x 向前、y 向左，单位米。动作接口预留 `[9,K,30,2]` 和 0.1 秒时间间隔；没有监督的模态保持 unknown。原始 yaw、测量速度及控制保留在高频日志中。

`source.json.assistance` 单独保存未来实际轨迹、自动候选和几何诊断，**不进入导出的 observation 或教师输入**。有带 epoch 的高频日志时从其提取参考线；没有时明确降为源 RGB-D 位姿轨迹，不伪称 30 Hz。

参考线检查使用已观测地图的膨胀障碍。unknown 是地图覆盖诊断，不是 RGB 可见性或路线标签有效性判断：人工审核通过的参考线允许经过 unknown，备注可选，也不因此关闭参考线 loss。`geometry_diagnostics` 保留中心线/车体范围的 unknown 采样点数量，`reference_review_policy` 记录批准策略。已知障碍冲突仍拒绝批准，需核对路线与地图。人工批准不等于未来全路径无碰撞证明；在线控制器的检查保持独立。旧批准记录不会自动升级，采用新策略须重新批准，后续依赖的 Route 修订也要重新审核。

首个 episode 的导出用途固定为 `regression`，`optimizer_updates_permitted=false`。它是人工基准，不同时作为训练样本和独立测试。整个任务批准后仍需根据实际使用反馈冻结格式，而非仅因为 JSON 可解析就宣布 M2 完成。

## 3. 统一命令入口

以下 Python 都使用 `/data/ticvla/envs/warehouse-io/bin/python`；不在 Isaac Python 中运行：

```bash
python -m ticvla.warehouse.dataset_pipeline --help
python -m ticvla.warehouse.dataset_pipeline extract --episode <raw/episode.json> --output <session/annotation>
python -m ticvla.warehouse.dataset_pipeline annotate --project <session/annotation>
python -m ticvla.warehouse.dataset_pipeline export --project <session/annotation> --output <new-export>
python -m ticvla.warehouse.dataset_pipeline validate --record <export/00000000.json>
python -m ticvla.warehouse.dataset_pipeline audit --root /data/ticvla/dataset --output <audit.json>
python -m ticvla.warehouse.dataset_pipeline convert --catalog <reviewed-catalog.json> --output <new-import>
python -m ticvla.warehouse.dataset_pipeline teacher --project <session/annotation> --config <teacher.json> --output <new-teacher-run>
python -m ticvla.warehouse.dataset_pipeline oracle --request <request.json> --solver <solver.json> --output <new-oracle-run>
python -m ticvla.warehouse.dataset_pipeline readiness --output <readiness.json>
```

### 教师

配置模板：[teacher.example.json](../configs/dataset_preparation/teacher.example.json)。填写实际 API model ID，密钥仅从指定环境变量读取。通过 [Responses API](https://developers.openai.com/api/docs/guides/images-vision) 发送因果图像、MapView、原始任务、旧 Route 和过去事件；每个时刻一次请求，最多 6 个不同关键帧，不自动修复失败。

当前实现用于**已批准人工基准上的教师对照**，未通过整段人工批准时直接阻止调用。`proposal.json` 的 Landmark/Local/Route 来自教师，waypoint 来自人工批准参考线；原文、实际模型、用量、耗时、来源哈希分别保存，不伪造费用或批准状态。第二个未标注 episode 的“仅审核参考线、逐轮审核教师 Route”流程尚需在首个基准完成后补齐，当前不宣称已通过自动标注验收。

### 数据审计和池

[pools.json](../configs/dataset_preparation/pools.json) 定义人工基准、静态语义路线、动态动作、外部动作、候选评分和辅助理解六个池，正式训练关闭。来源权重待合格数量确定后填写。

审计保留固定最多 10 个候选 episode/来源，包含失败项，不按成功结果重新筛选。核对 [catalog.example.json](../configs/dataset_preparation/catalog.example.json) 中的坐标、单位、时钟、外参、许可和任务匹配后才能导入。当前转换器仅支持经审计的轮式机器人 RGB/位姿，不补造深度、路线、九模态和评分。人类/四足数据的机器人化转换尚未实现。布局族先分组再切窗，重复原始轨迹和跨划分布局会拒绝。

### ALTRO 与候选评分

```bash
bash DynaNav/warehouse_loop/setup_altro.sh
```

独立 Julia 1.10.10、Altro 0.5.0，完整 [Project/Manifest](../configs/dataset_preparation/altro/) 和 Julia 下载哈希已锁定。不升级 Isaac、现有 Python 或 OCR 依赖。求解 worker 是 `run_altro.sh`，`solver.json` 的 `argv` 指向 `/bin/bash` 和该脚本的绝对路径。

输入严格限定为 `source_binding,t_obs,state,reference,static_circles,tracks,observation_frame`。`state=[0,0,0,v,w]`，`observation_frame=source_body_flu`。轨迹和障碍在同一源车体系中；动态 tracks 必须来自 `rgbd_tracking` 的至少两个过去时刻。独立评测器才可读取未来真值。

27 个 Lattice 初值 → 官方 AL-iLQR → 独立动力学、密集碰撞、速度/加速度、模态检查 → 分项排序。求解失败保持 unknown，不推断整个模态不可行。纯原地转向另标为 XY 动作接口不可充分表达。输出未经过动态真值与 Isaac 执行核验时，`quality_gt_ready=false`。

无障碍接口复现：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline oracle \
  --request /data/ticvla/outputs/dataset-preparation/oracle-interface-fixture/request.json \
  --solver /data/ticvla/outputs/dataset-preparation/oracle-interface-fixture/solver.json \
  --output /data/ticvla/outputs/dataset-preparation/oracle-interface-fixture/new-run
```

这是合成输入接口检查，**不能替代三个真实动态场景或算训练数据**。

### 九模态训练接口

`FlowTrainingAdapter` 使用部署 `FlowMatchingActionExpert` 的真实结构，稀疏候选 mask 控制 Flow Matching loss；VLM K/V detach，压缩器和动作侧可反传。checkpoint 严格保存/加载九模态结构，旧单候选路径保留。

`readiness --cache` 只接受标注、输入绑定和 VLM checkpoint 哈希一致的真实压缩前 K/V 缓存。没有缓存时明确阻塞，不拿合成特征替代。现阶段通过的是数值夹具上的梯度和加载接口；真实 VLM 文本/参考线 loss、TEXT/PREFILL 缓存及完整冻结边界仍待批准数据后验收。没有提供一键正式训练入口。

## 4. 本轮验证和剩余里程碑

详细机器报告：[dataset_preparation_20261014.json](validation/dataset_preparation_20261014.json)。本机完整工件在 `/data/ticvla/outputs/dataset-preparation/`。

| 里程碑 | 当前状态 |
|---|---|
| M1 人工工具 | Isaac 遥操作、命令过期停车、采集、回放和修订接口已测；manual-static-004 已完成用户批准及英文版本复查，剩余数据质量限制见文首报告 |
| M2 格式定稿 | v2 草案及兼容适配已实现；等待人工使用反馈后冻结 |
| M3 自动文本 | 教师真实 API 接口和人工基准门槛已实现；0 次实际调用，没有真实教师成功样本 |
| M4 审计/转换 | 本地目录盘点完成，DynaNav 105 个原始轨迹；元数据人工核验/合格转换尚未通过；SCAND/GND 为空 |
| M5 动态最小单元 | 官方 AL-iLQR 数值夹具通过部分候选；三个动态场景、未来真值核验和 Isaac 执行未完成 |
| M6 训练就绪 | 稀疏九模态 loss/压缩器梯度/严格加载接口测试通过；真实批准数据、真实语义缓存和完整多头 loss 验收未完成 |

验收日志明确区分测试夹具与专家样本。数据覆盖不足，训练未开始，不宣称模型或导航质量提升。

### 上一轮 Local 与短期相对运动

标注回放与导出的 observation 新增 `previous_local`；来自前一个已批准关键时刻，
不是上一个 RGB 帧，也不读取当前标注答案。在线导航使用上一轮成功提交的 TEXT Local，
PREFILL 不覆盖它。改写前一批准记录仍使后续标注失效，需重新审核。

`current_in_previous_body` 为 `inv(T_odom_previous) @ T_odom_current` 的平面位姿：
`x_m` 向前、`y_m` 向左、`yaw_rad` 逆时针；这是当前车体在上一 Local 源车体系中的位置与朝向。
它是净位姿变化，不是累计路程；上一车体在当前系的表示是该变换的逆，不能只对三个数取负。
来源图像、时间、版本、人工/模型来源及 age_s 一并保留。默认最大来源年龄 30 秒（工程初值）；
unknown 清除记忆，跨 odom epoch、非过去来源、过期或非法位姿标记 valid=false。

Local 始终是未核验的语义判断，不是定位 TF；仅历史推断用 `Inferred:` 前缀说明。
反复生成并不构成新的视觉确认，当前协议尚未独立追踪每条语义陈述的最初视觉证据，
30 秒阈值仅限制上一回答的源时间，不能保证递归语义误差不累积。
输入、教师调用与训练导出共用这份上下文，不增添额外 VLM 调用。
已有服务器需保存草稿后重启 replay 才加载新的 Python 输入逻辑；原始数据与既有批准文件不自动改写。

### 回放补记执行事实与停车标注

没有实时打事件标记的 episode 不需要为此重新采集。回放页面提供“回放补记事实”：
在对应源帧勾选确实已发生的 start、转向、target_visible、stopped、arrival_confirmed 等事实，
填写核验依据，并随本时刻人工批准保存。stopped 必须结合历史观测或运动记录核验，
不能仅凭发出停车命令或静态截图认定。页面输入信息显示 robot_state、有效性和 measured_facts。
只确认可从该时刻及以前数据得到的事实，不使用未来画面提前确认到达。

补记记录保存在批准版本的 replay_events，带源图像、时间、odom epoch、标注人、
来源哈希和说明，来源为 human_replay_review；不改写 source.json 或原采集事件。
只有当前及之前同 epoch 的批准事实进入 observation.execution.events 和导出／教师输入。
修改旧事实沿用批准版本链使后续标签需要重新审核；重复批准不会重复累加事件。

“现在应当停车”使用 current_subgoal.action=stop、status=active，不要求完成事件。
“已经到达并停稳、整体任务完成”才使用 completed，且需要目标可见、停车和到达证据。
不要为通过校验勾选尚未发生的事实，也不要把缺失动作监督转为停车动作 GT。

### 按实际任务验收（取代固定两次转向门槛）

整段最终确认不再要求两个 turn_completed 事件或单独 start 标记。原两次转向要求来自
示例任务，不适合单次转弯或直行任务。最终确认仍要求全部关键时刻批准有效、人工确认
原始任务完成，以及 target_visible、stopped、arrival_confirmed 完成事实。
原关键帧数量要求保持不变。转向事件保留用于回放与进度监督，数量只记为
recorded_completed_turns；程序不靠关键词推断任务应转几次，也不自动判定路线完成。
旧项目无需重采集或补造转向事件；重新启动回放服务后使用原批准记录继续最终确认。
