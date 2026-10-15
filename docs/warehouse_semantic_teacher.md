# 第二步：英文 Landmark / Local / Route 教师标注试验

本入口只请求三个语义字段。waypoint 和 reference line 来自示范轨迹／人工批准参考线，不送给教师，也不由教师预测。没有启动正式训练。最初各6帧的会话辅助试验已扩展为37＋17份密集提议，仍待人工审核；真实 API 调用尚未验收，不能将会话提议冒充 API 验收。当前调用方法见 [README](../README.md#人工与辅助标注第二阶段)，下一步优先执行[静态数据对齐计划](warehouse_dataset_alignment_plan.md)。

API入口目前仅接受最多6帧的 `semantic-teacher-pilot-v1`，会话导入支持整段 `semantic-episode-review-v1`。两者不可混用；整段API执行和批准提议的统一导出仍需接线。

## 两种上下文与标注用途

- `recursive`（默认）：第一轮使用原任务；下一轮使用上一次格式及规则检查通过的教师 Route。Local 使用带来源的上一教师提议及有效的相对运动。所有提议仍是 `approved=false`，递归试验不等于人工批准为 GT。任何非法响应或无证据完成会阻断后续递归，不补问、不回填人工答案。
- `gold_context`：只适用于已有人工基准，以当时人工批准的历史 Route/Local 为上下文，检查单步回答。它与自主递归有不同的输入分布，必须分别报告。

当前帧人工答案仅在 `evaluation_only`，不进入请求。当前及历史 RGB、因果 MapView、任务、过去执行事实属于输入。未来 RGB、未来轨迹、waypoint、参考线、原数据 CoT 和隐藏目标坐标均不进入请求。

输出使用 Responses API 的 JSON object 模式，并经过当前部署解析器及英文生成语法校验；不声称 JSON 模式能保证语义或全部字段正确。[官方图像接口](https://developers.openai.com/api/docs/guides/images-vision)、[结构化输出说明](https://developers.openai.com/api/docs/guides/structured-outputs)。每个源时刻一次请求，每个试验包最多6帧，无自动修复调用。

## 运行当前人工基准

使用 `/data/ticvla/envs/warehouse-io/bin/python`，不启动 Isaac，不修改人工批准记录。

先复制 `configs/dataset_preparation/semantic_teacher.example.json`，填写账户真实可用的 `model` ID、`base_url` 和已安全配置密钥的环境变量名 `api_key_env`。不要把密钥写入 JSON 或版本库。模型与凭据未配置时，保存请求准备报告，调用数为0，不产生伪造答案。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-prepare \
  --project /data/ticvla/datasets/manual-static-004/annotation \
  --indices 0 3 4 6 8 9 --context-mode recursive \
  --output /data/ticvla/outputs/teacher-manual-bundle

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-teacher \
  --bundle /data/ticvla/outputs/teacher-manual-bundle \
  --config /path/to/teacher-config.json \
  --output /data/ticvla/outputs/teacher-manual-run
```

`--prepare-only` 永不调用 API。递归模式下尚未得到上一轮回答的输入预览标记 `request_ready=false`，不是已经形成的真实递归请求；运行时重新构造实际上下文。

保存 `input.json`（完整问题、上下文、图像顺序/MIME/哈希）、`response.json`（API原始响应）、`raw.txt`、`proposal.json`、`comparison.json` 以及独立的 `geometric-supervision.json`。参考线来源与模型返回的语义字段分别标明。保留返回的实际模型名、usage、延迟；没有可靠价格信息时费用为空。

## DynaNav 迁移

仓库的 `DynaNav/configs/benchmark_full.yaml` 有5个 `num_people:0` 的仓库任务。例如 `episode_66`：

> Go straight to leave the current aisle, then turn right to approach the fire extinguisher cabinet on the wall.

这是仿真任务配置，不等于下载数据中同名轨迹的静态性证据。本地有30段仓库原始记录、230个重叠窗口；本轮固定选 `warehouse_14`，按去重后的时间均匀选6帧，不按教师成功筛选。该记录图像含多名行人，仅作为语义迁移试验。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-prepare \
  --dynanav-root /data/ticvla/dataset/DynaNav \
  --episode-id warehouse_14 \
  --output /data/ticvla/outputs/teacher-dynanav-bundle

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-teacher \
  --bundle /data/ticvla/outputs/teacher-dynanav-bundle \
  --config /path/to/teacher-config.json \
  --output /data/ticvla/outputs/teacher-dynanav-run
```

应先完成当前人工基准的真实模型接口验证，再执行外部数据真实调用。请求包可以提前准备。

适配规则：

- 原始 episode `instruction.txt` 是固定任务；不把逐帧改写的 instruction 或原 CoT 当作 GT/教师提示。
- 按原 episode 与时间戳去重重叠窗口，并检查30Hz源索引与发布的时间戳一致。
- CSV 与源相机矩阵独立重建对照；JSON 的未来 offset 与同一相机系重建结果对照。这个误差是数据表示的一致性，不是物理定位精度。
- 位姿标为 `camera_center_flu_not_robot_base`，不伪造车体外参。没有可靠机器人速度时保留无效 mask，不能把零填充值当停车证据。
- 没有深度就不提供 MapView。不同相机（此样例是鱼眼）保留原 RGB，不以针孔模型作投影。
- 发布的未来轨迹只保存为 `candidate_unreviewed`；空 future 不是已确认停车。静态性、任务一致性、机器人基座外参、碰撞尚未核验，`usable_for_robot_reference_training=false`。
- 外部数据没有独立 Landmark/Local/Route 真值，语义准确率为空。当前本地 SCAND/GND 无可用 episode，不编造迁移结果。

## 自动评测能证明什么

自动检查来源、格式、英文约束、因果时间、原任务和递归 Route、无证据 `completed`，并完整记录拒绝/未执行帧。

人工基准上统计类别和源帧匹配、当前子目标/action/status 一致性及文本精确匹配。人工只选择部分物体，因此类别匹配是标注一致性，不是目标检测精确率；Local/Route 词句不同也不直接意味着语义错误。完整语义正确性仍需独立审核，Oracle 自己不能作为自己的评测真值。

通过第二步需要真实 API 成功提议、真实递归 Route 正例、人工逐字段审核，以及另一 episode 的迁移结果。仅通过离线测试或生成请求包不构成通过。

上段是原自动 API 方案的通过条件。用户选择先使用订阅内交互辅助后，本轮完成语义提议、递归上下文和导入校验，API 项继续标为未执行；这不代表全部自动标注阶段已经完成。

当前准备产物：`/data/ticvla/outputs/dataset-preparation/semantic-teacher-pilot-20261015/index.html`。

## 会话辅助试验（2026-10-15）

当前结果：`/data/ticvla/outputs/dataset-preparation/session-teacher-20261015/index.html`。
在保存产物的机器上启动静态文件服务：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8773 \
  --bind 0.0.0.0 \
  --directory /data/ticvla/outputs/dataset-preparation/session-teacher-20261015
```

浏览器访问 `http://<服务器IP>:8773/`；本机使用 `http://localhost:8773/`。页面包含当前/历史图像、完整问题与上下文、原文、人工逐字段对照及参考线审核图。原文不会因审核发现问题而覆盖。

| 项目 | 人工 episode | DynaNav warehouse_14 |
|---|---|---|
| 固定时刻 | 1027、1280、1365、1534、1703、1788 | 0、339、678、1017、1356、1695 |
| 格式、来源、完成证据规则 | 6/6 | 6/6 |
| Route 递归回传校验 | 5次 | 5次 |
| 参考线与锚点 | 原人工批准值，未修改 | 原轨迹候选，未审核，不能直接作为机器人路线 GT |
| 人工 action/status 一致 | 各5/6 | 无独立人工真值 |
| 完整语义准确率 | 未评测 | 未评测 |

这次回答由当前 Codex 会话逐个查看图像后撰写，保存来源 `codex_conversation_assisted`。`input.json` 是每个时刻的输入材料，不是隔离模型调用的完整上下文。本会话以前已讨论过人工答案及后续场景，故 `context_isolation=false`、`independent_blind_evaluation=false`。本次没有自动 `/responses` 调用，没有独立的 API model ID、usage 或单次推理计时；这些字段保持空值。所有提议 `approved=false`，没有覆盖人工基准，也没有训练。

具体审核事项：

- 人工1027：Local 的 `shutter` 缺少充分证据，已标为待修订，不因格式合法而认定语义正确。
- 人工1365：当前和3秒历史图像均没有标牌，Landmark 留空。路口事实用于 Route，不能凭历史物体生成当前帧地标。
- 人工1534/1703/1788：比人工多选可见 sign，须审核选择策略；不能把“未在人类列表里”直接当作误检。
- 人工1703：人工为 `stop/active`，提议为 `wait/waiting`，因尚缺到达确认。需要统一“已停但等待验收”状态策略。约1.9 cm微小参考线原样保留。
- DynaNav1356/1695：原指令要求右转，左侧出现叉车，后续记录的相机累计偏航约 +1.215 rad；是需要核验的任务/轨迹一致性疑点，不直接认定文本错误。保留 `needs_clarification`，没有用未来轨迹改写原任务。
- DynaNav 最后一帧没有可靠停车/到达事件，不生成 `completed`。含行人的未来示范也不自动升格为静态参考线。

### 顺序导入与重放命令

`semantic-session` 是准备与导入工具，本身不会调用模型，也不能把订阅转换为 REST API。每次只准备一个输入，人工/会话完成回答后再导入；下一轮使用该提议 Route。失败原文归档并停止递归，没有自动补问。

```bash
# 新建目录，准备第一个输入；也可换成 dynanav-bundle。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-session \
  --bundle /data/ticvla/outputs/dataset-preparation/semantic-teacher-pilot-20261015/manual-bundle \
  --output /data/ticvla/outputs/my-session-review

# 阅读生成的 00000000/input.json 及其中引用的图像，准备回答文件后导入。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-session \
  --output /data/ticvla/outputs/my-session-review --response /path/to/answer.json

# 准备下一时刻；重启后可继续执行，同一个输入不重复创建。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_pipeline semantic-session \
  --output /data/ticvla/outputs/my-session-review
```

回答文件为 JSON 信封：`input_sha256`、`source_binding` 从该次 `input.json` 原样复制；`answer` 仅含 `landmark`、`local`、`route`。不能添加 waypoint/reference 字段。现有 `submitted-answer.json` 可用于导入重放；重放不是新模型推理。

逐帧保存 `input.json`、`submitted-answer.json`、`raw.txt`、`proposal.json`、`geometric-supervision.json`。全段结束后才生成 `comparison.json`，避免当前人工答案成为后续输入。画廊另存 `human-reference.json`、`review-note.json`、`input-contact.jpg` 和 `reference-review.png`；后两类审核材料不进入教师输入。

下一步先审核已标出的语义问题与状态规则，再使用未讨论过的 episode、隔离的调用上下文做独立验证。API 自动化仍保留为另一条可配置路径，本轮不扩成批量数据生产。

## 静态替换与参考线覆盖更新（当前入口）

本轮发现两个问题：下载的 `warehouse_14` 有行人，不能作为已核验静态数据；首版六时刻均匀抽样没有覆盖有限长参考线。人工首条线长约7.75m，34.8秒时车已离开参考线超过0.3m，而下一份回答在50.6秒才更新。

DynaNav benchmark 的仓库任务并非全部动态：`benchmark_full.yaml` 的 episode_61、66、71、76、81 均为 `num_people: 0`。这些 benchmark 配置不能直接证明同名下载轨迹无行人；本地采集默认配置为50名角色，旧录制确实可见移动人物。

新版浏览器入口：`/data/ticvla/outputs/dataset-preparation/static-refresh-20261015/videos/index.html`。服务仍使用8774端口。

| 场景任务 | 更新数 | 最长更新间隔 | 更新前耗尽参考线的区间 |
|---|---:|---:|---:|
| 人工灭火器任务 | 37（原6） | 5秒 | 0（原1） |
| 无行人叉车任务 | 17 | 5秒 | 0 |

第二段为Isaac Sim 5.0基础仓库重新渲染：保持原记录相机路径，固定场景时间，不创建行人，不运行物理控制。同步记录284帧RGB-D、源位姿和标定。任务改为接近可见黄色叉车，不沿用旧指令中与录像存在疑点的右转要求。它是新的静态相机回放；原轨迹来自动态录制，不因此获得静态最优路线或机器人可执行GT资格。新相机为960×540针孔，原始为鱼眼，文本全部根据新图像重新标注；不用于严格模型效果对照。

`reference_schedule.py` 的离线补帧规则：最长5秒；参考线余量低于1m或剩余25%（取较小值）时刷新；距线超过0.25m或航向变化20度时提前刷新；保留原标注和末帧。5cm以下停稳抖动不导致每帧刷新。全部源帧独立检查累计运动是否超出线长，而非只检查视频显示帧。本次两段均无耗尽区间，源位姿到活动参考线的最大距离分别约1.28cm和10.31cm，后者包含弯道离散化误差。这是覆盖诊断，不是导航精度指标。

原人工批准的6份几何标签保持不变；新增参考线从未来记录轨迹提取并标为未审核。54份文本来自本轮会话辅助标注，格式/来源校验通过，但未人工批准、无独立API调用，也不是实时InternVL推理。原始数据和旧视频均保留。视频以源时间切换，不含真实推理耗时；在线调度仍需把推理延迟计入提前刷新余量。

```bash
# 为人工任务生成密集审核输入，不生成或批准文本。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.reference_schedule \
  --episode /data/ticvla/datasets/manual-static-004/capture/raw/episode.json \
  --seed-session /data/ticvla/outputs/dataset-preparation/session-teacher-20261015 \
  --seed-group manual --output /data/ticvla/outputs/dataset-preparation/dense-review-new

# 已保存回答重放到新视频目录，无需再启动Isaac。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.episode_video \
  --session-root /data/ticvla/outputs/dataset-preparation/static-refresh-20261015/session \
  --output /data/ticvla/outputs/dataset-preparation/static-review-videos-new

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.video_server \
  --port 8774 --directory /data/ticvla/outputs/dataset-preparation/static-refresh-20261015/videos
```

静态采集脚本 `DynaNav/warehouse_loop/render_static_replay.py` 使用宿主Isaac Python，可从宿主仓库根目录执行：

```bash
PYTHONPATH="$PWD" /data/isaacsim/python.sh DynaNav/warehouse_loop/render_static_replay.py \
  --source /data/ticvla/dataset/DynaNav/DynaNav_data/warehouse_14 \
  --output /data/ticvla/outputs/static-camera-replay-new
```

实验目录中保存 `static-scene-audit.json`、Isaac日志、两段coverage报告、逐源帧输入/回答、几何标签、补帧清单和复现脚本。旧的6＋6视频仅保留作为问题复现版本。

## 按 episode 连续回放（旧的6＋6版本）

一个 episode 对应一个场景任务；本轮为 **2 段视频，每段 6 次结果更新**，不是 12 个场景。

入口：`/data/ticvla/outputs/dataset-preparation/episode-text-videos-20261015/index.html`。

- `manual-static-004`：灭火器任务，154.3 秒；从任务录制的第1027帧开始。
- `warehouse_14`：叉车任务，58.6 秒。
- 播放到每个标注源帧时，同时切换 Landmark、Local、Route、waypoint 和 ref_line；其余时刻保留上一份结果并显示来源年龄。每段末尾含2秒静止阅读时间。
- 右上轨迹图固定在任务起点坐标系，各次参考线经源位姿变换后显示。文字中的 waypoint 数值仍属于各自标注时刻车体系（DynaNav 为相机中心 FLU）。
- 这是已有同步 RGB/位姿与已保存回答的连续回放，没有重新推理，不代表实际推理延迟或新的 Isaac 执行。文本仍为待审核提议；几何标签与批准状态未改写。
- 每段保存 `updates.json`（6次完整结果）、`replay-frames.jsonl`（逐视频帧来源）、MP4、WebM及切换截图。页面按钮可跳转至每次更新。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.episode_video \
  --session-root /data/ticvla/outputs/dataset-preparation/session-teacher-20261015 \
  --output /data/ticvla/outputs/dataset-preparation/episode-text-videos-new

# 支持视频字节范围请求，浏览器可直接跳转；打开 http://服务器IP:8774/
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.video_server \
  --port 8774 \
  --directory /data/ticvla/outputs/dataset-preparation/episode-text-videos-20261015
```

## 逐 case 未来运动视频（旧的单时刻诊断）

12 段视频及浏览器索引：`/data/ticvla/outputs/dataset-preparation/trajectory-text-videos-20261015-v2/index.html`。
每段从自己的标注时刻播放到连续记录结束；使用同步录制的 RGB 与位姿，原始时间 1 倍播放、输出5 fps。不是重新在 Isaac 中运行控制器，也不是 AE 预测。

- 左侧：完整原图等比例显示，源图像索引、源时间和相对位姿。
- 右上：源时刻坐标系中的固定参考线、五个锚点、完整未来实测轨迹和当前回放位置；x 向上、y 向左，单位米。
- 右下：该 case 冻结的 Landmark、Local、Route 与子目标/status；这些不是后续每帧的新回答。
- 左下：waypoint 数值、ref_line 有效点数及标签来源。完整32点数组在 `case.json`。

灰线是用于审核的未来运动，绿线是已经播放的运动，橙线是标注参考线，青色为 waypoint，红色为当前位姿。未来运动不进入原教师输入。局部参考线长度有限；整段未来轨迹离开其范围不自动判定为错误。DynaNav 始终标为相机中心轨迹，未转换成已核验的机器人动作 GT。

人工1788和DynaNav1695没有连续后续记录，提供3秒静止展示并标注 `NO FUTURE RECORD`，不补造未来动作或停车成功。视频产生过程没有修改文本、参考线或人工批准状态。

```bash
# 重新生成到一个新目录，无需启动 Isaac 或模型。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.replay_video \
  --session-root /data/ticvla/outputs/dataset-preparation/session-teacher-20261015 \
  --output /data/ticvla/outputs/dataset-preparation/trajectory-text-videos-new \
  --fps 5

# 浏览器打开 http://服务器IP:8774/
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8774 \
  --directory /data/ticvla/outputs/dataset-preparation/trajectory-text-videos-20261015-v2
```

每个 case 包含 `replay.mp4`、`replay.webm`、`poster.jpg`、`last-recorded.jpg`、`case.json`、`pose-track.csv` 和 `replay-frames.jsonl`。CSV 的参考线距离只是几何诊断，不是导航准确率；JSONL 将每个编码帧绑定到真实 RGB 哈希、位姿与源时刻，静止展示帧单独记录。MP4 使用 H.264/yuv420p，另提供 VP9 WebM 以兼容缺少 H.264 解码器的浏览器。
