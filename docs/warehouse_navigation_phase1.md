# 自然语言导航与递归 Route：第一阶段

本实现直接接收 `instruction`，不由程序按关键词拆任务。原始指令固定保存；一次 InternVL TEXT 输出的结构化 Route 成为下一轮 `remaining_route`。格式合法、进度正确、参考线正确、实际运动和任务完成分别报告。

## 启动

在仓库根目录运行；脚本使用 `/data/ticvla/envs/warehouse-realtime/bin/python`，通过原宿主桥接启动 `/data/isaacsim` 的 Isaac Sim 5.0。

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --instruction "沿通道走到尽头，左转，然后前往72/4R标牌旁停下。" \
  --output "/data/ticvla/outputs/navigation/instruction-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

自然语言入口默认 `reference-only`：运行真实 VLM、原 Decoder、上下文压缩和 Flow 诊断，车辆不执行模型命令。浏览器地址由 `NAVIGATION_VIEW` 打印，默认 `http://127.0.0.1:8766`。

显式添加 `--execution-backend geometry-baseline`，几何控制器才消费模型子目标；添加 `--execution-backend model` 则仅允许通过参考线一致性及候选检查的模型轨迹控制车辆。没有静默回退。旧的 `--task-mode landmark|explore|follow` 保留，默认仍是 `model`，与 `--instruction` 互斥。

倒退诊断使用：

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --instruction "倒退5 m。" --execution-backend reference-only \
  --output "/data/ticvla/outputs/navigation/backward-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

“是否理解为倒退”仍取决于模型；程序不使用关键词重写模型子目标。任何模型生成的 backward 子目标都禁止物理执行，其参考线诊断不调用地图碰撞检查。自然语言 `reference-only` 默认值也防止误解后的其他动作被执行。

`--scene-person` 仅在仿真中增加一名行人，不进入 VLM 任务输入。`--scene-layout junction-fixture` 创建独立的物理路口测试场景，不向导航传入布局或真值。`--budget-seconds` / `--budget-distance` 是运行预算，耗尽记录为 `budget_exhausted`，不是完成。`--timeout-seconds` 对应独立任务超时；采集达到步数上限记录 `capture_limit_reached`。

旧地图须传 `--map` 和明确重叠图像 `--map-anchor-image`，并通过几何验证。不会默认跨 episode 坐标相同。

## 协议与状态

五个标签及顺序保持不变：

```text
<landmark>[{"image_id":"源图像ID","category":"sign"}]</landmark>
<local>空间关系或 unknown</local>
<Route>{"remaining_instruction":"剩余指令","current_subgoal":{"action":"explore"},"status":"active"}</Route>
<waypoint>[]</waypoint><REFLINE>
```

以上是协议示例，不是实验模型答案。Route 状态为 `active/waiting/needs_clarification/completed`。动作及参数：

| action | 参数 |
|---|---|
| approach_landmark | target_text、category、description 至少一项 |
| advance_to_intersection | ordinal：仍需经过的路口数 |
| advance_to_corridor_end | 无 |
| turn | direction：left/right |
| relative_motion | direction：forward/backward；distance_m；origin：subgoal_start |
| explore | 无；预算来自配置 |
| follow_person | description |
| wait / stop | 无 |

语法约束只约束可用字段、动作和参数类型，不给模型预选动作或目标。解析仍接受旧纯文本 Route，但自然语言入口要求新协议。旧文本不会被自动转换为可执行子目标。

waypoint 是五个有序空间锚点，单位米，源车体系 x 向前、y 向左；不含时间，不是轮速。无法表达路线可返回 `[]`。连续线始终由实际处理 `<REFLINE>` 的 hidden state 送入原 Decoder 得到，没有 waypoint 插值替代。

原始 Decoder 数组、mask、置信度均保存。非空 waypoint 检查到线距离 0.30 m、端点差 0.50 m 与顺序；这些是初始工程检查阈值，不是导航精度标准。TEXT 空 waypoint 对非空 Decoder 线标为不一致。PREFILL 不做文本 waypoint 检查。

合法 Route 可与不可执行的同轮参考线一起原子提交。下一轮仍读新 Route，控制器不能使用该参考线，也不能搭配上一轮旧线。任务、版本、请求顺序、来源年龄和 odom 不符时拒绝提交。

执行事实源于 odom、地图和真实检测，包括子目标锁定起点、累计路程、转角、路口进入/离开事件和目标观测。改写措辞、递减路口序号或距离不能重置同一子目标原点。没有相应执行证据的 completed 只保存为模型声称，不进入任务完成状态。当前实现不自动证明任意复合自然语言任务已全部完成；需要查看原指令、执行事实和实际 Route。

## 文件与代码

| 文件 | 职责 |
|---|---|
| `ticvla/warehouse/navigation/route_protocol.py` | Route schema、prompt、解析和参考线诊断 |
| `navigation/execution.py` | 实测执行事实、锁定起点、路口去重、证据检查 |
| `navigation/core.py` | 固定任务、递归 Route、同轮原子快照、恢复 |
| `navigation/route_geometry.py` | 从已观测占用提取保守通道/路口候选、几何子目标控制 |
| `navigation/runtime.py` | 异步推理、单写入者、地图/任务衔接与真实控制发布 |
| `navigation/phase1_replay.py` | 六组×三轮真实模型离线测试 |
| `navigation/phase1_validate.py` | 原仓库/独立场景 × 四任务 × 两后端，及静止倒退诊断 |

每个在线输出目录含：

- `config.json` / `environment.json`：配置、依赖来源。
- `navigation/route-history.jsonl`：每轮实际输入 Route、新 Route、版本、执行事实及验证。
- `navigation/execution-events.jsonl` / `state.json`：事件与可恢复工作状态。恢复时不复活旧命令或过期张量，必须重新推理。
- `navigation/vlm/*/request.json,response.json,result.pt,snapshot.npz`：真实输入输出、原始张量及阶段耗时（以实际目录为准）。
- 新请求另存 `prompt.txt`、`answer.txt`、`audit.json`；有 Decoder 输出时另存 `reference.csv`，包括全部点及原始有效性 mask，便于直接阅读。失败输出同样保留，未运行的头不生成假数组。
- `navigation/committed.json`：同一次 Route 与快照元数据的原子提交标记。
- `navigation/flow/`、`metrics.jsonl`、`executed-commands.jsonl`：候选、拒绝、真正发给 Isaac 的命令。
- `capture/raw/`、`navigation/map/`、`report.json`、`index.html`：采集、独立地图、报告与浏览器视图。

图中橙色为原 Decoder 线，青色为 waypoint，浅蓝为 Flow 候选，绿色为选中路径，紫色为实测轨迹。存在某条线不代表可执行或完成任务。

## 复现验证

```bash
OMP_NUM_THREADS=2 /data/ticvla/envs/warehouse-realtime/bin/python \
  -m unittest discover -s tests/warehouse

/data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.navigation.phase1_replay \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --output /data/ticvla/outputs/navigation/phase1-replay-new

/data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.navigation.phase1_validate \
  --seconds 24 --output /data/ticvla/outputs/navigation/phase1-isaac-new
```

离线三轮使用明确标记的执行事实夹具，RGB 保持固定；不是物理运动或实际跟踪证据。在线测试分别生成新 episode，实际任务计划完全来自 VLM，测试脚本不会注入子目标。24 秒是本次排错试验预算，不能用于宣称长任务不可完成。失败保留，不通过人工计划替换。

本阶段未训练。原有单测包含小型训练代码回归夹具，不产生部署导航权重或占用之前约定的训练预算。

### 少量重跑与输入审计

不需要重跑整个 Isaac 矩阵。以下命令只运行三个诊断任务，每个两轮真实 TEXT；每轮仍只有一次问答：

```bash
/data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.navigation.phase1_replay \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --cases landmark explore reverse --rounds 2 \
  --prompt-style original \
  --output "/data/ticvla/outputs/navigation/phase1-small-$(date +%Y%m%d-%H%M%S)"
```

`--prompt-style compact` 仅用于离线对照，不改变在线默认 prompt。每次使用新的输出目录。

`audit.json` 记录原始任务、工作 Route、图像来源、真实 token 数和计时。`prompt.txt` 是原始完整 prompt，视觉占位在模型输入中由真实图像 embedding 替换；`answer.txt` 是模型原文，不包含程序解析与补造文本。

TEXT 新计时口径：`first_token_wall + decode_wall = native_generation`。首 token 时间从生成调用前开始，包含 prefill 与第一次 token 选择；decode 墙钟时间包含其后的生成循环、语法筛选和结束同步。`grammar_wall` 与 `prefill_gpu/decode_gpu` 是嵌套诊断，不能再累加。`final_query` 是补处理末尾查询标记的前向；`heads` 包含 Decoder、K/V 打包、压缩和数值检查。旧 `reference_and_compressor` 聚合字段保留兼容。PREFILL 沿用原计时，没有 TEXT 的首 token/decode 字段。模型加载、排队、prompt 构建及文件写入不包含在文本生成时间中。

工程隔离检查复用测试入口，包含正确前进与转向子目标的运动学反馈，以及真实控制 watchdog 的取消、过期、新障碍与坐标会话检查。这些夹具不连接车辆，不作为真实 VLM 或 Isaac 任务成功，也不向部署入口增加人工子任务列表。

## 当前边界

- 原始 instruction 与递归 Route 的职责已分离，完成状态需证据。
- 几何控制为保守工程对照；通道分支依赖局部已观测自由连通性，未验证复杂拓扑的鲁棒性。
- 地标按 UID、文字或类别精确查询，多匹配暂停；任意视觉属性的指代解析尚未实现。
- 跟随只支持唯一可见行人的初次自动绑定，丢失后遵守原连续性门限，不自动换人；长时间失踪仍需显式重新确认。
- 未训练 Decoder、压缩器、Flow 的输出质量不因接口通过而得到验证。
- 真实测试结果见 [第一阶段实验报告](warehouse_navigation_phase1_results.md)。
