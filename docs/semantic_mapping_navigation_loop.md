# 语义建图与导航标注的因果闭环

标注职责、完整协议、教师/学生模板与近期验收顺序见 [VLM 标注与输入输出规范](vlm_annotation_and_io_spec.md)。当前共用的是证据构建及详细 RULES；“教师详细、学生短提示”的 SYSTEM 分离仍待实施，不把历史闭环结果当作新模板质量验收。

## 当前输入与输出

当前模型输入格式为 `plain-evidence-v2`：[精简格式说明](vlm_landmark_local_prompt.md)。来源、版本与完整变换保存在外部 `audit_context`，`user_text` 保存实际发送的简短文本，`context` 为结构化记录。后文早期复验使用旧输入，不能视为新 prompt 已通过模型效果验证。

统一视觉输入由 `ticvla/warehouse/semantic_input.py` 定义，教师 API、本地 InternVL 训练/推理使用同一输入投影和图像顺序；完整候选表与语义记录只留在审计侧：

1. 当时可用的历史 RGB 和当前原始 RGB。
2. 上一时刻已提交的语义地图：占据底图＋地标位置、类别、可用 OCR 文字、tentative/身份状态。
3. 当前 RGB 的检测框与实例 ID 叠图，及其类别列表，不附框坐标和置信度表。

保留原始 RGB，检测叠图是附加图像；不把 mask 外区域涂黑。这样不会抹掉通道、遮挡和文字线索。全部图像沿用同一个视觉编码器，没有新增独立 MapEncoder。真实 mask 用于投影、关联和 OCR 归属，框/ID 用于让 VLM 理解检测建议。

仍采用已确认的紧凑输出 `landmark: [category...] / local / route`。一次回答兼顾登记对象和导航关系，不再额外问一次建图问题。**类别输出会放通该类别的全部检测实例，并非逐实例 ID 筛选**。未检测到的类别记录失败，不补造 mask；旧的独立主动建图实例筛选入口保留作兼容对照。

2026-10-28 更新：统一输入的 prompt 版本为 `candidate-landmark-local-v2`。有候选输入时，Landmark 只能从 `current_candidates.allowed_categories` 选择，并同时满足当前 RGB 可见、有同帧分割 mask；候选为空时输出 `[]`。Local 优先表达相对可辨识地标的位置和朝向，保留有依据的编号；历史推断标 `Inferred:`，未确认 OCR 不当作确定身份。详见[输入对照及简化示例](vlm_landmark_local_prompt.md)。这是生成要求，不是格式解码保证；旧结果保持原样，新 prompt 尚未经过真实模型复验。

## 严格时序

```
当前 RGB → 同帧 YOLO-E 候选
上一轮语义地图（仅较早观测）→ 对齐当前源车体系
两者＋原任务＋上一轮 Route/Local＋相对运动 → 一次教师/VLM
→ 校验文本 → 本轮 Landmark 对检测类别准入
→ 同帧 mask＋深度＋源位姿 → 关联/入图 → OCR 追加文字
→ 下一轮输入
```

输入地图在调用前冻结。本轮入图结果不能回填本轮请求，最终地图不能倒灌早期帧。首次语义层为空；占据底图可使用首次标注之前的原始几何观测。所有位置使用当前源车体系，x 前、y 左、米；机器人位姿来自 odom，不来自 VLM Local。

样本继续是 schema v2；`observation` 增加可选 `semantic_input` 和 `perception_candidates`，`mapview.landmarks` 带来源与状态。旧人工批准数据不改写，新请求、原文和地图写到独立目录。文本提案未自动批准，也没有启动训练。

## 离线标注闭环复现

以下命令使用现有人工 demo 的 10 个关键帧。每轮语义更新只发生在关键帧；占据底图读取该时刻之前的全部采集帧。这不是完整 5 Hz 在线建图或实时性验收。

```bash
# 在仓库根目录执行；输出目录必须是新的。
DISABLE_MODEL_SOURCE_CHECK=True /data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop prepare \
  --bundles \
    /data/ticvla/outputs/stage1a/oracle-comparison-20261021/bundles/manual-0 \
    /data/ticvla/outputs/stage1a/oracle-comparison-20261021/bundles/manual-1 \
  --weights /data/ticvla/outputs/landmark-realtime-20261012/engines/v8s/model.engine \
  --output /data/ticvla/outputs/stage1a/semantic-loop-new/perception

/data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop run \
  --prepared /data/ticvla/outputs/stage1a/semantic-loop-new/perception \
  --config /data/ticvla/outputs/stage1a/qwen-tiers-20261027/configs/plus-thinking-2048.json \
  --credentials /home/xinglinlu/.config/llm/credentials.env \
  --output /data/ticvla/outputs/stage1a/semantic-loop-new/plus

/data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop audit \
  --output /data/ticvla/outputs/stage1a/semantic-loop-new/plus
```

`prepare` 真实执行固定词表 YOLOE-v8s TensorRT FP16 与 CPU PP-OCRv5 mobile；缓存与哈希锁定。不同教师复用这些原始检测结果，但拥有独立 SQLite 地图，不复用其他教师的准入/文字绑定/Route。`run` 执行有预算约束的真实 API；2026-10-29 起，格式失败或不具备证据的完成声明只拒绝该帧，后续沿最近有效上下文继续，不做自动修复；API、预算及来源/存储异常仍停止。Local 仍按真实来源年龄过期。下一阶段采用 [Plus 固定配置](qwen_plus_annotation_profile.md)，历史停止结果保留。凭据内容不写入产物。

`audit` 不调用模型，检查父响应哈希、上一轮输出/下一轮输入地图版本、过去时间边界、图像哈希以及数据库重启恢复。

## 导航入口

`run_navigation.sh` 默认 `--semantic-input-mode unified`，`legacy` 可恢复旧输入对照。本地 TEXT worker 在同一源帧上准备检测候选，然后只调用一次 VLM；后续语义 worker 重用同一批 mask，OCR 检查全部候选的归属，仅向被 VLM 选中的成功实例追加文字。PREFILL 不执行这次额外候选准备。

本轮未重跑 Isaac 物理导航，也未重新验收新增图像带来的本地 VLM 延迟；几何/控制仍使用原有独立调度。在线地图是已提交的历史证据，异步 OCR 尚未完成的读数要等下一次请求才能出现。

## 2026-10-27 复验

产物：`/data/ticvla/outputs/stage1a/semantic-loop-20261027/`，总览 `index.html`，分别见 `plus/index.html`、`max/index.html` 和逐帧 `input.json / raw.txt / mapping/frame.json / map-after.json`。

- Plus：10/10 轮完成，18 次有效实例入图/更新，最终 10 个地标记录；派生展示层保留 4 个 sign 和 2 个灭火器候选，4 个普通货架隐藏。它们全部仍为 tentative。9 个后续请求实际包含历史语义记录。
- Max：2 轮完成，第 3 轮 Local 82 字符超过 80 字符上限，原文保留；后续 7 轮不再调用。不能算完成 10 帧闭环。
- 13 次真实 Qwen 请求估算 ¥0.687494（Plus ¥0.281714、Max ¥0.405780）；包含失败响应，沿用冻结的北京端点价格与共享预算账本，非实际账单。GPT 未调用。
- Plus 第 1365 帧没有当前检测，但引用地图的灭火器位置；第 1449 帧仍把历史目标写成当前 Landmark，并过早认为左转完成。没有 mask 的类别不会产生新地标，但其文本监督仍需修订。
- `02/4R` 和 `0214R` 不满足子串包含关系，当前规则拒绝其编号，保留位置。`01/4R`、`03/4R`、`04/4R` 有候选读数，未宣称唯一身份确认。
- Plus 在第 1703 帧仍要求继续接近，第 1788 帧有完成证据时才输出 DONE；旧对照在第 1703 帧曾过早 DONE。旧对照用人工历史上下文，新实验用自身递归输出，且增加了候选叠图，不能把变化全归因于语义地图。

验收结论：输入接口与实际地图反馈闭环通过，数据库重载通过；**标签准确性未全通过，未自动批准，不能报告泛化准确率提升**。100 项相关测试通过；没有报告车辆闭环导航或实时性能通过。

## 按 Agent.md 整理完整任务审核（2026-10-27）

上一轮 Plus/Max 结果已按仓库 [Agent.md](../Agent.md) 重新整理到：

```text
/data/ticvla/outputs/stage1a/semantic-loop-20261027/review-v1/
```

打开 [审核总览](http://localhost:8789/review-v1/) 或该目录 `index.html`。其中包含全部 10 个关键帧 × 2 个模型的输入、原文、逐字段 Agent 复核、地图/OCR 结果、人工审核导出按钮，以及一个约 152.6 秒的整段任务对照视频 `episode.mp4`。Max 未运行的 7 个位置不补造答案。当前为教师标注，Waypoint 和 Decoder 预测标为不适用。

实际 API 图像编码已从原始输入复现，并核对全部 13 个请求的哈希；原输入输出未改写。此次整理没有 API 调用，旧费用与本次新增费用分开统计。Agent 复核不是用户批准，人工模板默认 `approved=false`。

在新目录复现整理（不重新调用模型）：

```bash
/data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop_review build \
  --source /data/ticvla/outputs/stage1a/semantic-loop-20261027 \
  --reviews /data/ticvla/outputs/stage1a/semantic-loop-20261027/review-decisions-v1.json \
  --output /data/ticvla/outputs/stage1a/semantic-loop-20261027/review-new

/data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop_review video \
  --output /data/ticvla/outputs/stage1a/semantic-loop-20261027/review-new

/data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.semantic_loop_review audit \
  --output /data/ticvla/outputs/stage1a/semantic-loop-20261027/review-new
```

输入 `review-decisions-v1.json` 是本次 Agent 视觉/日志复核及原响应哈希，不是自动评分器。换实验时需重新审核，不能直接沿用这些 verdict。视频按源帧时间对齐，不按 API 实际返回时间对齐，不能据此声称在线导航或实时性能通过。原推理的精确请求发送/返回墙钟和完整代码 dirty diff 未记录，报告保留这一缺失。
