# VLM 人工审核与评测报告标准

版本：`vlm-human-review-v1`。

2026-11-12 特定 demo 协议补充：用户要求仓库两任务的 Oracle 输出采用 `rgb-pixel-waypoint-v1`，waypoint 是源 RGB 中的 `(u,v)` 像素坐标，必须经同帧 optical-Z 深度和标定投影。此协议只覆盖显式标记的新 demo，不将已有车体系米制标签重新解释为像素。PCHIP 是工程跟踪曲线，不是 Reference Decoder 输出。离线固定回放、Codex 会话代标注和实际闭环分别报告；完整说明见 [RGB waypoint demo 规范](docs/rgb_pixel_waypoint_demo.md)。

2026-10-28 补充：标注职责、输入输出与教师/学生模板设计见 [VLM 标注规范](docs/vlm_annotation_and_io_spec.md)。证据输入已使用 `plain-evidence-v2`；教师详细 SYSTEM 与学生短 SYSTEM 的分离尚待实施，审核不得将草案写成已运行配置。

本文件规定后续 Agent 在评测 VLM、比较标注模型、生成训练标签审核材料时的**标准交付格式和判定方法**。它是审核报告规范，不是发送给被评测模型的额外 prompt，也不改变模型输出协议。人工审核说明可用中文；模型原始输入输出按实际语言保留，当前英文协议不自动翻译或润色。

## 1. 审核单位与两类评测

**一个 episode 是一个场景中的一次完整移动任务。报告必须覆盖该任务全部预定关键帧，按时间排列，不能只挑成功帧或最后一帧。** 同一任务不能因为分成多个请求文件夹而被拆成多个独立成功案例。

| 模式 | 审核的模型输出 | Waypoint / Reference line |
| --- | --- | --- |
| `teacher_annotation`：补充中间监督标签 | Landmark、Local、Route | 不要求教师生成。若由未来示范轨迹派生，单独作为监督辅助展示，注明来源，不计为教师输出，不送给教师 |
| `system_evaluation`：系统模型评测 | TEXT 的 Landmark、Local、Route、waypoint，以及真实 Decoder 输出 | 必须保存并展示真实 waypoint 和参考线；空值、无效 mask、失败均保留，不能用 GT 或插值路线替代 |

系统运行 PREFILL_FOLLOW 时不生成文本：对应字段标为 `not_applicable`，展示实际参考线及快照来源。若显示上一轮 TEXT，必须标明它的源帧、时间和年龄，不得冒充本轮生成。

单帧调试可以作为附录，但不能据此给出整个移动任务通过的结论。

## 2. 实验封面与完整覆盖

报告开头必须给出：

- 模式、run ID、episode ID、场景/数据来源、完整原始 Instruction、任务开始和结束时间。
- 模型实际 ID、checkpoint/revision（可获得时）、配置与代码版本；思考开关、生成上限、格式约束、实际 API 或本地后端。
- 检测器、OCR、词表、地图配置、相机标定和 odom 来源；是否使用理想仿真 odom。
- 上下文类型：人工审核历史、模型自身递归历史，或真实在线上下文。明确车辆由人工示范回放、几何对照还是模型控制。
- 全部预定关键帧清单与选择规则、总数、实际发起/返回/解析有效/人工审核/批准数量。说明运行前是否冻结清单。
- 使用历史地图还是当前几何图、历史图像数量、是否包含检测叠图。与本规范不同的旧输入如实标记，不能事后补图后声称模型当时见过。

关键帧需覆盖起步、接近路口、转弯前/中/后、目标出现、接近、停止，以及该任务中实际发生的丢失/歧义/恢复。未发生的事件不强造。

系统评测须检查相邻推理帧的间隔与上一条参考线剩余覆盖；若已驶出参考线而尚无新输出，单独记录覆盖缺口。没有位姿或参考线数据时写“无法核验”。已经冻结但覆盖不足的集合应报告不足，不能仅增加容易成功的帧。

## 3. 每个关键帧的固定展示顺序

### A. 当前帧身份

显示 `frame_index`、`image_id`、源时间、距任务起点时间、请求 ID、请求/结果到达时间，以及距上一关键帧的时间与相对位移/转角。模型选择源帧、检测源帧和地图源时间分别记录。

### B. 模型实际输入

必须在审核页直接展示图片缩略图并可打开原尺寸，不能只列磁盘路径。

| 输入项 | 展示要求 |
| --- | --- |
| 当前原始 RGB | 保留未经 mask 涂黑的原图，标源帧与尺寸 |
| 分割/检测 RGB | 展示模型实际收到的框、ID 或 mask 叠图和类别列表；当前未发送的框/置信度表放到审计侧，准确注明到底是检测框还是像素 mask |
| 历史 RGB | 展示所有实际传入帧及时间偏移；源 ID 和完整相对位姿可另列审计信息，不得用当前图代替缺失历史 |
| 输入语义地图 | 展示调用前冻结的地图，包含占据底图和实际传入的地标层；说明图例、尺度、坐标系、机器人方向、地图版本和观测截止时间 |
| 地图地标表 | 当前输入标 not_provided；若展示完整 UID/类别/OCR/位置/来源/身份记录，必须放在审计侧。旧实验确实传入表时按原请求展示 |
| Instruction 与递归上下文 | 原始 Instruction、上一轮 Route、上一轮 Local 与相对运动；Local 来源/有效性、执行事实和机器人状态等额外元数据按实际是否发送分栏 |
| 完整请求 | 可展开的实际 system/user prompt、图像顺序、预处理/尺寸/编码、文件及实际传输哈希（可获得时） |

**严格区分“送入模型的图”和“仅供审核的图”。** 当前实现给 VLM 的补充图是检测框/ID 叠图，真实像素 mask 另供投影与审核。报告可增加彩色 mask 叠图，但必须标“审核辅助，未输入模型”。不能把这种附加图列为实际输入。

没有某项输入时显式标 `not_provided`。只有占据图的旧实验应写“占据图，无语义层”，不能改称语义地图。首次空语义地图也是合法状态，应展示空层而非预加载最终地图。

`plain-evidence-v2` 输入将模型上下文与审计信息分开：`input.json.user_text` 是实际发送的简短文本，`context` 是其结构化记录。Progress/执行事件与文字地标表不进入模型，`audit_context` 保存原始 ID、哈希、版本、完整位姿与事件来源，**不发送给模型**。审核页可展示两者，但不能把审计信息描述成模型实际见过的输入。RGB 用年龄标记，`d0` / `L1` 只用于图文对应，不是数据库身份。

模型输入只包含当时可用的信息。GT、未来轨迹、未来图像、隐藏目标坐标和人工本轮答案只能进入隔离的审核/监督面板。

教师和学生不要求 SYSTEM 字面相同：教师可有详细标注规范，但不能获得学生不可用的未来证据。评测需记录实际教师/学生模板文本及版本/哈希，检查学生训练与部署的模板、字段语义、图像顺序和预处理是否一致。当前教师和本地模型共用证据投影不等于串行化完全相同。新旧 SYSTEM 对照先固定观测，再单独比较自身递归结果；模型不一致的历史不能混称为同输入对照。

### C. 模型实际输出

同时提供**未经修改的原文**和**解析后的逐字段表**，不能只给自然语言总结。

教师当前协议为 JSON，必须保留原始 JSON；可另作标签式展示，但注明“程序渲染，不是另一段模型生成”。系统 TEXT 使用实际外层标签与大小写：

```text
<landmark>当前可见类别的 JSON 数组</landmark>
<local>模型实际生成的 Local</local>
<Route>模型实际生成的剩余指令</Route>
<waypoint>五个有序坐标或 []</waypoint>
<REFLINE>
```

以上只是布局说明，不是有效答案示例，不能写入结果充当模型输出。

当前紧凑协议的审核要点：

- Landmark 最多三个简短英文类别字符串，不要求模型输出 image_id。程序绑定的源 ID、规范化类别及原始类别另列，不能混入模型原文。
- Local 当前上限 80 字符；Route `compact-v2` 当前上限 600 字符，使用自然语言字符串或有证据支持的 `DONE`，不要求 `current_subgoal/status`。旧协议按其冻结版本单独检查。
- 系统 waypoint 是请求源时刻车体系内五个有序空间锚点：单位米，x 向前、y 向左；不表示动作时间或轮速。合法空列表不等于模型给出了正确路线。
- `<REFLINE>` 是查询标记，连续参考线由 Decoder 输出，不是教师或语言模型直接生成的一串参考线坐标。
- 教师的 Landmark/Local/Route 与程序派生的 waypoint/ref_line 必须分别标记 `field_sources`。教师未被要求生成 waypoint 时填 `not_applicable`，不能用 `[]` 冒充一次空路线预测。

2026-10-29 起，按 [Plus 标注配置](docs/qwen_plus_annotation_profile.md) 隔离坏帧后继续时，须展示 `context_recovery` 的实际上下文来源、跳过帧及 Local 过期状态；不能称为紧邻上一帧的完整递归。请求覆盖与有效标签覆盖分别统计，失败帧仍在分母中。

解析失败、超长、拒答、截断、错误引用仍展示原文和精确原因。不同于解析失败，合法但不正确的内容应归入语义错误。格式约束不证明语义正确；人工修订另存版本，不能覆盖原文或静默修复后记为模型通过。

### D. 本轮下游结果与下一轮反馈

显示以下证据，尚未执行的分支明确写 `not_run`：

1. 模型类别如何匹配当前检测类别；别名规则、未匹配类别、对应实例 ID 和 mask。
2. 每实例的选中/未选中、投影接受/拒绝、关联 UID、过滤前后点数及理由。类别准入会放通该类全部候选，不能描述为模型选中了每一个具体实例。
3. OCR 区域、原文、置信度、候选承载实例、最终归属、冲突及处理结果。编号不凭单次读数升级为确认身份。
4. 更新后地图与本轮新增/更新/拒绝的地标；输入地图版本、输出地图版本、下一轮是否实际复用该版本和父响应。
5. 现有地图、历史 Local 或检测建议有错时，展示该错误是否影响后续输出，不能把这些输入当成真值。

系统评测额外展示：原始 Decoder 数组、有效 mask、置信度、参考线 BEV 与 RGB 投影，以及 waypoint 叠图。超出相机视野的投影点标为不可见，不移回画面。GT 路线仅以不同颜色在审核侧展示。

分别评价文本锚点与 Decoder 的方向、尺度、顺序和端点一致性；有检查阈值时记录配置。waypoint 为空而 Decoder 非空的 TEXT 输出须报告不一致。不能因为 Decoder 未训练，就隐去其真实结果或把它算成文本错误。

若运行控制，再展示实际控制来源与车辆状态；仅评测 Local/Route 时，禁止把文本描述当作已发生运动。

## 4. 人工逐字段评价

统一 verdict：`pass`、`partial`、`fail`、`unscorable`、`not_applicable`。`partial` 不计为完整通过；未审核使用 `review_pending`，未运行使用 `not_run`。每个评价需有证据引用和具体理由，不只给分数。

| 维度 | 必须检查的内容 |
| --- | --- |
| 输入/来源 | RGB、检测、深度、位姿是否同源；地图与历史是否因果；当前帧是否误读本轮或未来入图结果 |
| 格式 | 对照该运行实际协议解析；数量、长度、坐标与来源是否合法 |
| Landmark | 当前可见性、类别正确性、登记价值、重要目标遗漏；地图中存在不等于当前可见；背景重复结构不能仅因静止而优先登记 |
| Local | 左右/前后、编号与承载物是否对应；依据当前 RGB、历史还是地图；只靠记忆的推断是否适当标注不确定性 |
| Route | 保留原始最终目标、剩余动作顺序和距离/路口计数；仅按已发生事实缩减进度；不提前删掉转向或宣布完成 |
| waypoint（系统 TEXT） | 与当前子目标、源坐标系、方向/距离及连续参考线一致；空路线有无合理依据 |
| Decoder（系统） | 实际 mask/有限数值、方向、尺度和路线一致性；与语言 waypoint 的错误分开归因 |
| 地图反馈 | 类别准入、mask、深度、关联、OCR 与下一轮复用是否正确；重复事件是否增加证据 |

语义判断允许同义表达，不按文字与人工答案逐字相同判对错。检测漏检不自动判为 VLM 选错；VLM 输出无 mask 也不必然表示物体不存在。OCR 未读出与 VLM 幻觉必须分开。没有独立真值或视觉证据不足时用 `unscorable`，不得猜测精确 IoU、三维误差或目标身份。

完成判据以本次任务约定为准：人工仓库 demo 的距离阈值不可直接套用 R2R/RxR。`DONE` 必须有本任务要求的到达/停止等证据；“开始转弯”不能当成“完成转弯”，发出停车命令不能代替已停车事实。

建议统一错误码，并可追加版本化新码：

```text
INPUT_SOURCE_MISMATCH / FUTURE_INFORMATION / INPUT_NOT_REPRODUCIBLE
FORMAT_INVALID / OUTPUT_TRUNCATED / SOURCE_REFERENCE_INVALID
LANDMARK_NOT_VISIBLE / LANDMARK_CATEGORY_WRONG / LANDMARK_MISSED / LOW_VALUE_SELECTION
LOCAL_UNSUPPORTED / LOCAL_DIRECTION_WRONG / MAP_MEMORY_AS_CURRENT_VISIBILITY
ROUTE_GOAL_LOST / ROUTE_STEP_DROPPED / ROUTE_PROGRESS_PREMATURE / PREMATURE_DONE
DETECTION_MISS / MASK_INVALID / CATEGORY_UNMATCHED
OCR_MISREAD / OCR_CONFLICT / TEXT_WRONG_CARRIER
DEPTH_INVALID / PROJECTION_REJECTED / ASSOCIATION_AMBIGUOUS / WRONG_MERGE
WAYPOINT_INVALID / WAYPOINT_ROUTE_MISMATCH / DECODER_REFERENCE_INVALID
MAP_FEEDBACK_BROKEN / REFERENCE_COVERAGE_GAP / PROVIDER_FAILURE
```

可同时记录多个错误及其传播链。推测的根因须标为推测，区分日志已证实的原因。

## 5. 整段任务总览与视频

审核页必须提供按时间排列的完整关键帧表，默认显示所有行，包括失败和未运行行：

| 帧 / 时间 | 当前图 / 候选 / 输入地图 | Landmark | Local | Route | waypoint / Decoder | 入图 / OCR | 评价 / 原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 每个预定关键帧一行 | 可直接查看图片 | 原文或展开内容 | 原文 | 原文 | 教师注明不适用 | 数量与详细记录 | 分字段 verdict |

另给逐帧详情、前后 Route 变化和错误首次出现位置。后续纠正、错误传播、恢复时间应有对应帧，不能只展示“最终结果正确”。

**有连续图像和位姿时，一个 episode 交付一个视频**：按记录的实际移动顺序回放，在对应推理帧更新 Landmark/Local/Route；系统评测同时更新 waypoint/参考线。冻结期间显示上一输出的源时间和年龄，不能无标记沿用。应说明是按源帧时间对齐的离线审核，还是按实际结果到达时间回放；前者不能用于证明在线低延迟。

视频不能替代全部关键帧表和原始文件。只有稀疏关键帧时，可以输出明确标注的关键帧幻灯片；必须说明缺少连续记录，不能称为连续运动回放或伪造中间帧。

同一场景有多个模型时使用同一关键帧清单和并排展示。固定人工上下文与各模型自身递归上下文分别统计；递归后的地图不同是闭环效应，不能宣称后续输入完全相同。旧版占据图输入与新版语义地图＋候选图对比属于整体方案对照，不能将全部收益归因于地图。

## 6. 汇总指标和结论

分别汇总，禁止合成一个不透明的“成功率”：

- 任务覆盖：预定关键帧、实际请求、返回、格式有效、审核过、批准、失败、未运行的数量及原因。
- 字段质量：每字段 `pass/partial/fail/unscorable/not_applicable/review_pending` 数量；语义通过率明确可评分分母，另列 `pass / 全部预定适用帧` 的端到端有效覆盖率，不能靠漏掉失败帧提高结果。
- 递归质量：目标保持、错误进度/提前完成、地图记忆误作当前可见、首次错误与传播帧数、恢复情况。
- 地图结果：接受/更新事件数、去重后的实例数、可用文字/冲突数、错误归属与合并、静态/身份状态；不能把 18 次更新写成 18 个独立地标。
- 系统路线：非空 waypoint、有效参考线、方向/尺度/一致性以及覆盖缺口；格式有效不等于可执行。
- 资源：模型请求数、实际 token、思考 token（可获得时）、价格来源、含失败请求的费用；VLM、检测、OCR、投影/入图和发布分别计时，给 p50/p95 与样本数。

API 仅有整次请求耗时时，标为“请求往返耗时”；没有服务端细分不能伪称 prefill 或 decode 耗时。样本少时说明 p95 的局限。审核耗时另计，成本分清总请求成本与审核合格标签成本。

最终结论分别回答：输入来源是否可信、格式接口是否通过、各语义字段是否合格、地图反馈是否发生、系统参考线是否正确、运动/实时性是否实际验收。未测项写未测；不以接口贯通替代正确性，也不将十帧表现外推为全数据集准确率。

## 7. 标准交付目录与审核记录

```text
review-run/
  index.html                    # 完整任务总览、模型对照、逐帧详情
  REPORT.md                     # 结论、证据、失败归因和限制
  experiment.json               # 模式、模型配置、协议、代码/数据来源及哈希
  keyframes.json                # 预定全部关键帧，包含未运行帧
  report.json                   # 计数、指标、用量、费用和测量口径
  human-review.json              # 人工逐字段判定，默认未批准
  episode.mp4                   # 连续回放；缺数据则记录缺失原因
  frames/<image_id-safe>/
    input.json                  # 实际上下文、图像顺序、路径与哈希
    prompt.txt                  # 完整可读请求文本，不含凭据
    input-images/               # 原图、历史、实际候选图、输入地图
    response.json / raw.txt     # 原始响应及原始生成文本，包括失败内容
    parsed.json                # 解析结果；失败时记录原因，不补造字段
    review-only/               # 未输入模型的 mask、GT、未来轨迹等
    mapping/                   # 入图事件、OCR 绑定、输出地图及版本
    reference.npz              # 系统实际 Decoder 数据；教师不适用
    checks.json                # 格式/来源/一致性检查和本帧状态
```

允许现有实验沿用其他目录布局，但总览须链接对应文件并说明映射，不要求为改名复制大型产物。HTML 不得包含密钥；文件不存在、未运行或不适用时记录状态与原因，不生成空白伪结果。交付前检查本地链接和图片可打开。

`human-review.json` 每项至少包含下列字段；以下为**未审核模板**，不是示例模型答案：

```json
{
  "run_id": "<run>",
  "episode_id": "<episode>",
  "image_id": "<source>",
  "model": "<actual_model_id>",
  "mode": "teacher_annotation",
  "response_sha256": null,
  "status": "review_pending",
  "fields": {
    "landmark": {"verdict": "review_pending", "reason": "", "evidence": []},
    "local": {"verdict": "review_pending", "reason": "", "evidence": []},
    "route": {"verdict": "review_pending", "reason": "", "evidence": []},
    "waypoint": {"verdict": "not_applicable", "reason": "Teacher does not generate geometry", "evidence": []}
  },
  "error_codes": [],
  "suggested_revision": null,
  "approved": false,
  "reviewer": null,
  "reviewed_at": null
}
```

系统评测需增加 Decoder 的判定项，并按 TEXT/PREFILL 设置适用性。一个帧的人工批准不能自动批准同 episode 其他帧。Agent 自己的视觉复核标为 `assistant_review`，不能冒充用户人工批准。修订文本另存并绑定原响应哈希；训练标签批准记录与评测原模型成绩分开保存。

## 8. Agent 最终回复模板

最终回复至少提供：

1. 本次评测模式、完整场景任务、模型与关键帧覆盖 `已返回 / 预定总数`。
2. 审核总览链接、完整报告、视频或不可生成原因；直接展示至少一组有代表性的实际输入/输出对照，不能只有目录。
3. 各模型格式通过、语义待审/通过/失败、地图反馈与系统路线状态。
4. 至少列出实际发现的关键失败帧与原因；没有证据时明确不能判定。
5. 是否已人工批准、能否作为训练标签，以及未完成项。不得在没有人工审核时称“全部标注正确”。

后续新增或重跑报告按本标准执行。历史报告是否满足本标准应另作检查，不能因新增此文件便声称旧报告已自动补齐。

相关实现与说明：[联合输入构建](ticvla/warehouse/semantic_input.py)、[教师输入](ticvla/warehouse/dataset_tools/teacher.py)、[结构化解析](ticvla/warehouse/structured_io.py)、[Route 协议](ticvla/warehouse/navigation/route_protocol.py)、[闭环复验](docs/semantic_mapping_navigation_loop.md)。

2026-10-29 教师 v2 更新：新标注 Local 必须简洁且最多90字符。实际 user prompt 另发送 `Completion evidence: confirmed|unverified`，由已有完成事件计算；完整事件表仍仅供审计。审核应按实际运行配置显示限制，不能把旧实验80字符规则改称90字符，也不能把离线重新校验写成新模型调用。

2026-11-04 教师v5：Route只描述剩余导航动作及停等要求。审核区分路径中的地标引用与当前位置描述，记录重复Local、已完成步骤回流、历史总结过长、无依据Waiting等问题；不因出现inside/left等位置词就自动判错。v5SYSTEM缩短不等于效果已验证，历史v4按其实际prompt归档。
