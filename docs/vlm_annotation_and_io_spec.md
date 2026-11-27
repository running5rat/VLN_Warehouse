# VLM 标注方案与输入输出规范

**2026-11-20 下一版本设计补充：**[落地计划](implementation_plan_20261118.md)确定VLM／Reference使用前视与原有语义地图上下文、AE使用多方向观测，学生输出两个RGB像素Waypoint并保留连续Reference Decoder。下文五锚点等内容记录已有协议；新旧协议通过版本和显式转换区分，运行代码、旧标签与checkpoint尚未迁移。语义教师仍仅生成Landmark/Local/Route，不新增Waypoint标注调用。

更新：2026-10-28。适用范围：静态导航语义标注、语义地图反馈、Stage1a 训练准备及后续车端 TEXT 推理。本文汇总当前约定；历史实验按当时冻结的协议解释，不据本文改写历史输入、响应或成绩。

**当前教师选择：**用户暂定 `qwen3.7-plus`（思考预算 2048，输出上限 8192），使用已冻结模板及“坏帧隔离、后续继续”策略。见 [Plus 配置与标注条件](qwen_plus_annotation_profile.md)。尚未批准为无人审核教师。

2026-10-29 实验补充：教师调用已支持通过冻结配置显式指定 `teacher_instruction`，保存模板版本与哈希。当时三模型实验使用各自归档的冻结模板（不是现行第3.2节v5）；默认教师与车端提示未切换，学生短 SYSTEM 尚未实施。见[人工场景对照报告](manual_teacher_review_20261029.md)，三配置均有中断，不能把新模板标为质量验收通过。

## 1. 当前实现与下一步设计

| 项目 | 当前状态 | 本轮约定 |
| --- | --- | --- |
| 数据记录 | schema v2：observation / supervision / provenance | 保持格式，监督缺失用 mask 表达，不补造标签 |
| 模型证据输入 | `plain-evidence-v2`，教师与本地 VLM 共用输入投影 | 保留简短文本、原始/历史 RGB、检测叠图和语义地图图像 |
| Landmark 输出 | `current-categories-v1`，类别字符串数组 | 不生成 image_id；程序绑定当前源帧 |
| Route 输出 | `compact-v2`，英文剩余指令或 `DONE` | 不生成 current_subgoal/status；旧协议显式兼容 |
| SYSTEM | 教师与车端仍共同附加详细 RULES | **待实施：详细教师规范与短学生模板分离** |
| 教师选择 | 已有 Qwen 分级小样本结果；标签待审核 | 当前只继续比较 Qwen；GPT 计划 pending，GLM 历史结果保留 |
| 训练 | 人工 demo 已完成有限拟合诊断 | 新模板需重新验证；未批准标签不进入正式训练 |

本次文档更新不等于短 SYSTEM 已接入代码、模型已完成训练或新提示已通过质量评测。当前 `build_prompt()` 同时供本地训练和推理使用；它仍附加详细规则。实现位置见本文末尾。

## 2. 教师、学生与程序的分工

**教师的 prompt 不必与学生逐字相同；学生训练与部署应使用相同的固定模板。** 三者应共享字段含义、坐标约定和可用观测边界。

```text
详细教师规范 + 当时可用观测 → Landmark / Local / Route 提议
                                      ↓ 人工审核
未来静态示范轨迹 → 审核参考线 → 同路线五锚点
                                      ↓ 组装监督
短学生提示 + 当时可用观测 → SFT 文本目标 + Reference Decoder 监督
                                      ↓
车端使用同一短学生模板，运行解析、来源及执行检查
```

- 教师负责语义标签，不提供米制几何真值；教师的解释、推理过程、审核说明不作为学生输出目标。
- 学生学习字段含义、固定格式和导航关系。短提示仍保留任务与协议入口，不要求当前基础 InternVL 零提示掌握自定义协议。
- 程序负责源帧绑定、类别准入、语法/长度检查、时序检查、地图更新和执行安全。格式约束不能保证语义正确，SFT 也不替代这些检查。
- 教师可以获得更详细的标注规则，不能因此获得学生当时不可用的未来图像、隐藏目标坐标、当前人工标准答案或未来轨迹。
- 车端使用短 SYSTEM 的收益需实测。输入 token 影响 prefill 和上下文开销，不等于逐 token 的输出 decode；图像 token、历史缓存及生成长度另计。

## 3. VLM 的实际输入内容

### 3.1 共用证据格式

以下是逻辑模板，尖括号为占位符，不是固定答案。实际图像顺序与预处理以保存的请求为准：教师当前先发文本再发图像，本地接口先拼图像占位再拼文本；短模板实施时应明确冻结各端顺序，不能声称现有序列化完全相同。

```text
Instruction: <original English instruction>
Previous Route: <previous accepted remaining instruction>
Previous Local: <previous valid Local, or unknown>
Motion since previous Local (<age> s): x_m=<x>, y_m=<y>, yaw_rad=<yaw>
Current detectable categories: <current mask-backed category names, or none>

<image>Historical RGB, 3 s ago</image>
<image>Historical RGB, 2 s ago</image>
<image>Historical RGB, 1 s ago</image>
<image>Current raw RGB</image>
<image>Previous committed semantic map, rendered in current body frame</image>
<image>Current RGB with detector boxes and instance IDs</image>
```

| 输入项 | 来源及含义 |
| --- | --- |
| Instruction | 原任务始终保留，不被 Route 覆盖。当前标注与训练采用英文；场景编号原样保存 |
| Previous Route | 第一次用原始 instruction 初始化；后续来自上一轮合法输出。人工历史对照使用批准版本并单独标记 |
| Previous Local | 上一轮有效的位置关系描述；不是世界坐标或定位真值。过期、odom epoch 不一致或不可用时为 unknown，省略其相对运动 |
| 相对运动 | `inv(T_odom_previous_local) @ T_odom_current`：当前车体在上一 Local 源车体系中的位姿，x 前、y 左、米，yaw 逆时针弧度；年龄不是固定的 1 秒 |
| 检测类别 | 从当前同帧 mask 候选产生；用于限制 Landmark，不是模型必须全部选择的答案 |
| 原始/历史 RGB | 保留通道和文字线索；缺失帧明确标不可用，不伪造历史。不输入详细 rgb_history 元数据表 |
| 检测叠图 | 当前给 VLM 的是框/ID 叠图，不是逐像素 mask 图。真实 mask 留给投影、OCR 归属和审核 |
| 语义地图图像 | 此前实际观测并接受的语义层，加截至当前时刻可用的几何层；在当前源车体系重绘。可显示实际 OCR 文字及 tentative 状态，非预加载真值地图 |

不向模型发送 `Progress`、`execution` 事件表、`arrival`、`velocity_mps_radps`、文字 `Map landmarks` 表、详细 `candidates` 框/置信度表，以及 batch/image ID、哈希、完整 TF 和版本号。保留 `Current detectable categories` 这一简短类别列表。教师 v2 另增加一行 `Completion evidence: confirmed|unverified`，只由截至当前帧记录的 target_visible、stopped、arrival_confirmed 三类事件共同决定；缺少任何一类均为 unverified，不使用未来轨迹、空参考线或末帧位置推断。完整事件表仍不发送。按需的用户硬约束、跟随目标观测和上一参考线属于专用接口，不能混入本轮静态教师实验而不记录。

这些数据可继续存在于 observation 和审计记录，不意味着全部进入 prompt。移除 Progress 与文字地标表是输入设计选择，不表示车辆无法在线生成测量事实或地图记录。地图图像仍可携带因果的历史语义信息。

### 3.2 教师 SYSTEM：用户指定精简版

新标注使用 `qwen_plus_teacher.v6.json`，版本 `v07-teacher-user-supplied-v6`。配置原样保存用户给出的文本，包括连写词与标点；文档代码块仅去掉行尾空格。v1至v5配置与历史实验保持不变。v6尚无真实API质量验证，模型/思考/输出预算不变。

```text
Return only an English JSON object with exactly landmark, local and route.

landmark: Up to three distinct useful object categories, copied from Current detectable categories and supported by current RGB and detections. Rejecthistory-only objects. Use [] if none; no IDs or explanations.

local: State current position and heading relative to identifiable landmarks, at most 90 characters including spaces and punctuation. Use supported readable identifiers; mark uncertain OCR as tentative and memory-only localization as Inferred:. Past map evidence may help even if landmark is []. Use unknown when unsupported.

route: Give only remaining navigation commands: where to go, turn, stop or wait; at most 600 characters. Do not describe current position, heading, scene layout or completed actions. Keep Previous Route unchanged when still appropriate.An unseen instructed target may remain a search goal, but do not invent its location or directions.Use exactly "DONE" only if Completion evidence: confirmed AND the entire original task is complete. Otherwise give the remaining or waiting command.

Use English letters, digits, spaces and only these punctuation characters inside strings: ',''.'. No symbols, emoji or Markdown.
Maps and previous Local may be wrong; gray cells are unknown. Motion is current body in the previous Local frame: metres, x forward, y left, yaw counterclockwise radians. Map uses current body frame.
```

注意原文标点限制与`Inferred:`、含`/`的编号存在冲突；当前只更换prompt，兼容解析器的既有字符范围未收窄，不声称已实现仅逗号/句号的强制解码。Route只描述剩余命令，Local描述当前位置；路径里允许使用地标指代位置，不用位置关键词正则误杀正常路线。

[本版token来源与费用报告](qwen_plus_prompt_cost_v6_20261104.md)。这是教师提示更新，不是学生训练/部署提示切换。

### 3.3 学生 SFT / 车端 SYSTEM：短模板草案

以下为**待实现、待对照验证**的模板。选定后同版本用于 SFT 和车端推理，不在训练时使用长规则、部署时悄悄删除。

```text
Update navigation from the observations and previous state. Use English.
Return only <landmark>[categories]</landmark><local>relation or unknown</local>
<Route>remaining instruction or DONE</Route><waypoint>five (x,y) pairs separated
by semicolons, or []</waypoint><REFLINE>.
Landmark uses current detected categories; Local describes landmark-relative pose.
Waypoints use the source body frame: metres, x forward, y left.
Use unknown or [] when unsupported. End at <REFLINE>.
```

Local/Route 长度、类别数量、字符语法仍由协议校验；详细的标注优先级通过审核样本学习。是否完全去掉格式模板属于后续消融，不是默认方案。PREFILL_FOLLOW 是独立模式：处理查询但不生成这些文本，不将普通静态样本当成跟随监督。

## 4. 输出格式与字段含义

### 4.1 教师模型原始输出

必须是三个小写键的 JSON，下面仅演示格式和有证据时的措辞，不是实测结果：

```json
{
  "landmark": ["sign"],
  "local": "By sign 01/4R, facing toward sign 02/4R.",
  "route": "Continue to the aisle end, then turn left and approach the fire extinguisher."
}
```

- `landmark`：最多三个英文类别，每项最长 40 字符；同类多实例只列一次，不拼接标牌编号。规范化空白和大小写后去重。编号由 OCR 与实例绑定；Local 可引用有证据的编号。
- `local`：非空英文字符串，最多 90 字符；证据不足使用 `unknown`。它是关系描述，不生成 TF，不覆盖 odom，也不把猜测提升为绝对定位。
- `route`：非空英文剩余指令，最多 600 字符；完成声明为大写 `DONE`。没有 `current_subgoal/status`，没有人工子任务树。目标不可见不等于完成；不能因某次运动开始就删除整个动作。
- 当前部署语法使用受限英文 ASCII 字符；标点应使用普通英文标点。实际可接受字符以 `response_grammar()` 为准，不能只通过 JSON 解析就视为合格。
- 不允许额外输出 waypoint、ref_line、image_id、解释或审核结论。失败、截断、超长保留原文，不裁剪修复后计为模型通过。

### 4.2 学生完整 TEXT 输出 / 训练文本目标

外层标签顺序和大小写固定；实际序列化连续连接标签。下面的五个坐标仅示范语法，不是任何样本的 GT：

```text
<landmark>["sign"]</landmark><local>By sign 01/4R, facing toward sign 02/4R.</local><Route>Continue to the aisle end, then turn left and approach the fire extinguisher.</Route><waypoint>(0.5,0);(1,0);(1.5,0);(2,0);(2.5,0)</waypoint><REFLINE>
```

waypoint 表示当前局部路线的五个有序空间锚点：请求源时刻车体系、米、x 前/y 左；不是带时间的动作、轮速或独立于参考线的第二条路线。无有效路线时用 `[]`。训练标签取自批准参考线的 20/40/60/80/100% 弧长位置；真实推理则是学生生成的预测，必须与 Decoder 输出分别保存和比较。

`<REFLINE>` 是查询标记，后面不由教师输出一串参考线坐标。系统读取实际处理该标记后的 hidden state，经 Reference Decoder 输出参考线及有效性；不把文本 waypoint 插值当成 Decoder 的真实结果。

参考线监督是独立的 `[32,2]` 米制数组和有效前缀 mask，默认采样间距 0.25 m，末端短段按现有提取规则处理。`valid`、`valid_empty`、`unlabeled` 分别表示有效路线、确认空路线和缺失监督；不能互换。真实轨迹中的原地转向保留 yaw/控制证据，不把 XY 重复点直接视为停车完成。

程序将教师三字段和独立几何监督组装为完整训练文本，记录 `field_sources`。**组装文本中的 waypoint 不是教师生成的**；旧 image_id/category 记录和结构化 Route 只走显式兼容适配，原件保留。

## 5. 标注流程与地图递归

### 5.1 人工基准与几何标签

1. 在静态 Isaac 仓库中人工遥操作完成一个真实可行任务，保存 RGB-D、标定、源位姿及实际事件。一个 episode 是完整场景任务。
2. 默认约 10 个关键时刻并覆盖转弯、目标出现、接近和停车，不要求每张图标注；自动提取密集时刻须检查参考线覆盖，不能上一条参考线已用完仍不更新。
3. 从后续静态示范轨迹提取局部参考线，去除停顿重复点、变换到源车体系；人工审核、修订后派生同路线 waypoint。保留原轨迹、候选和批准版本。
4. 按时间顺序标 Landmark/Local/Route；修改早期批准 Route 后重新检查后续依赖。未来轨迹只在标注辅助/监督侧显示，不允许把未来事实写进当前语义标签。
5. 逐帧批准并确认整段。原始采集、修订和来源哈希不可被新实验覆盖。首个人工 episode 是回归基准；有限拟合诊断不作为泛化评测。

采集/回放命令见 [人工准备说明](warehouse_dataset_preparation.md) 和 [README](../README.md#人工与辅助标注第二阶段)。首个静态任务不标注 AE；缺失动作标签不能填成停车 GT。

### 5.2 教师单次请求与语义地图反馈

```text
冻结当前 RGB-D/位姿 → 运行同帧检测、保存 mask/OCR
                           ↓
此前已接受语义层 + 截至当前几何层 → 当前车体系 MapView
                           ↓
原始 instruction + 上轮 Route/Local + 相对运动 + 图像 → 一次教师请求
                           ↓
保存原文 → 解析/来源/类别检查 → 本轮语义提议
                           ↓
选中类别的同帧实例 → 深度投影/关联/入图 → 源实例 OCR 绑定
                           ↓
下轮使用新 MapView、合法 Route 与有效 Local
```

- 输入地图必须在请求前冻结。本轮新增语义不回填同轮输入，最终地图不倒灌早期帧。第一轮没有历史语义时显示空层，不能填入最终地标目录。
- 类别数组会放通该类别的所有候选，程序仍逐实例检查；不能把它说成 VLM 精确挑选了每个实例。无 mask 不补造地标，失败明确归因于检测/准入/投影/关联/OCR。
- OCR 文字按源实例绑定，不能复制给邻近货架；身份确认与静态确认分开，新地标保持 tentative。历史文字可能出错，模型引用它不自动证明定位正确。
- 递归实验使用模型自己上一轮合法输出，不用人工答案偷偷修复；2026-10-29 起，单帧输出校验失败仅隔离该帧，后续使用最近有效 Route/Local 与已提交语义地图继续；超龄 Local 失效，断点与失败保留在分母中。API、预算、来源或存储异常仍停止。格式合法仍是待审核提议。配置与例外见 [Plus 固定配置](qwen_plus_annotation_profile.md)。
- 固定人工历史对照与自身递归实验分别报告。不同模型的后续 Route/Local/地图会分化，因此只在共同冻结输入上比较单轮能力，在独立地图上比较闭环误差传播。
- 当前帧普通几何更新、检测和审计信息可保存，但不因此全部发送到 prompt。Local 不负责估计地图 TF，空间变换仍使用传感器位姿。

可复现的模型环境、准备和闭环运行命令见 [语义地图闭环说明](semantic_mapping_navigation_loop.md)。新 prompt 必须开新运行目录，不能将旧响应当作新模板结果；本次不发起 API 请求。

### 5.3 训练导出与监督边界

| 命名空间 | 保留内容 | 是否直接送模型 |
| --- | --- | --- |
| observation | 图像/地图路径、源绑定、位姿、任务、历史 Route/Local、检测和在线事实 | 只通过输入白名单与模板投影，不整体序列化 |
| supervision | 审核语义文本、独立参考线/waypoint、有效 mask；缺失动作状态 | 作为 loss 目标，不能进入用户观测 |
| provenance | episode/布局/来源、人工或教师信息、修订、哈希、审核状态 | 不进入 prompt |

当前静态语义训练只监督已启用的文本和参考线分支；未标注动作不贡献 loss。训练 teacher-forcing 的前缀不能包含未来轨迹；完整目标文本中派生 waypoint 属于监督序列。Decoder 既要看 teacher-forced 诊断，也要看自由生成后真实查询状态，避免只在标准答案上下文下报拟合成功。

原始教师回答、人工修订、组装文本分别保存。审核导出不是自动批准训练；先确认样本状态、来源分组、监督 mask 和 manifest 权限，再进入训练。

## 6. 近期实施顺序与验收

1. **短模板实现与冻结（下一项）**：教师规范和学生 SYSTEM 分开，固定图像顺序与预处理，保留 schema v2 和原数据；保存模板文本及哈希。现有长提示基线原样保留。
2. **同输入对照**：先用人工 demo 全部 10 帧比较本地模型长/短提示，分别检查格式、Local/Route、waypoint/Decoder；教师独立验证新规范。模型自身递归再单独运行，不能将此阶段混入固定输入对照。
3. **短模板训练诊断**：用相同短模板完成有限 SFT、保存/重载、自由生成检查，验证冻结边界、文本/参考线 loss 与未标注 mask；旧模板诊断通过不等于新模板通过。
4. **Qwen 教师审核与选型**：继续用完整 episode 比较质量、递归错误、人工修订量和每条审核合格标签成本；旧结果只作历史参照。当前 GPT 计划 pending，不自动恢复五模型付费比较。
5. **首批数据**：既定 R2R 120 + 英语 RxR 80 个训练 episode；已准备批次有 3,458 个待语义标注时刻。几何回放已准备不等于全部具有新语义地图/检测输入，先审计并补齐所需输入，再做小批审核后扩展。不要用占据图冒充语义地图。
6. **正式 Stage1a**：批准首批标签后，开展原初始化基线、轨迹/参考线监督、加入中间语义监督的对照；各组统一 split、评测接口和数据范围。扩展至 2,000 episode 需首批验收，不默认启动。动态 Oracle、FMAE、Scorer、RL/DAgger 继续暂缓。

按布局族和原始轨迹/episode 先划分再切窗；同一路线的不同指令、外观变体和重叠窗口不能跨 split。官方验证/测试集不用于教师生成训练标签。人工单 episode 的拟合和反复分析过的回归样本不能被称为独立泛化验证。

对照至少报告：全部预定帧覆盖、格式/来源、Landmark 可见性、Local 依据、Route 目标保持及提前完成、地图反馈、系统 waypoint/Decoder 一致性。计时区分 prefill、decode、请求往返与人工审核；只有 API 往返时间时不能补报服务端阶段耗时。成本包括失败请求和可获得的思考用量，真实账单与估算分开；不用旧提示均价冒充新提示实测成本。

## 7. 交付和复现记录

每个完整 episode 提供 `index.html`、全部关键帧表、逐帧实际输入图片与 prompt、未经修改的输出、失败原因及人工逐字段审核；有连续图像时提供一个 episode 一个视频，在对应源帧更新输出。具体标准见 [Agent.md](../Agent.md)。

每次请求至少保存：

- `input.json`：`instructions`、实际 `user_text`、结构化 `context`、有序图像路径及哈希；另存不发送模型的 `audit_context`。
- 原始 `response.json`、`raw.txt`、解析结果和失败原因；标明是否真正调用、是否截断和实际模型配置。
- 监督几何、字段来源及审核记录；教师输入界面与未来轨迹审核辅助明确隔离。
- 本轮地图事件、输出地图、下一轮父响应/地图来源，运行配置和代码版本。不得把 raw.txt 重写成润色后的人工答案。
- 拆分实施后需新增清晰的教师/学生模板版本与哈希，关联 checkpoint 和训练 manifest；不能仅凭共享一个 `input_format` 就宣称训练部署一致。

实现索引：[输入投影](../ticvla/warehouse/compact_input.py)、[教师规则](../ticvla/warehouse/dataset_tools/teacher.py)、[教师请求/组装](../ticvla/warehouse/dataset_tools/semantic_annotation.py)、[训练与推理共用 prompt](../ticvla/warehouse/vlm.py)、[Landmark 协议](../ticvla/warehouse/landmark_protocol.py)、[Route 协议](../ticvla/warehouse/navigation/route_protocol.py)、[格式解析](../ticvla/warehouse/structured_io.py)。

操作和历史结果：[人工链路](warehouse_dataset_preparation.md)、[教师试验](warehouse_semantic_teacher.md)、[Qwen 对照](qwen_teacher_comparison.md)、[地图闭环](semantic_mapping_navigation_loop.md)、[Stage1a 命令与报告](stage1a_training_acceptance.md)、[输入简化历史](vlm_landmark_local_prompt.md)。

### 2026-10-30：Route 引用必须有观测依据（v4）

Route 对物体、地点及空间关系的描述，应有当前 Local、输入图像或已观测地图中至少一种证据的支持。不能用三者均无法支持位置的物体或地点作为已定位的导航参照。Local 没提到但图像或地图支持的参照仍可使用。

Previous Route 仍然合理且有依据时可以原样保留，不为改写而改写。原指令中的未观测目标可以继续作为寻找目标，但不能虚构其位置或方向，也不能因为不可见而删除未完成要求。是否完成子步骤由证据决定，DONE仅用于有证据的整项任务完成。

v4替代尚未实测的v3“必须基于当前Local”的过窄要求；不设置物体名称黑名单，不强制Route文本每轮变化。Local仍必须简洁且最多90字符。此次未调用API。
