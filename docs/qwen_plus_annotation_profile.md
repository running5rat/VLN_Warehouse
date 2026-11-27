# Qwen Plus 下一阶段标注配置（2026-11-04）

新标注使用用户指定的 `qwen-plus-annotation-v6`，[配置](../configs/dataset_preparation/qwen_plus_teacher.v6.json)与[哈希锁](../configs/dataset_preparation/qwen_plus_teacher.v6.lock.json)。SYSTEM原样保存，未自动修正连写词或标点。模型与预算不变，v6尚未真实调用API验证；历史v1至v5及其报告保持原样。

[本版token来源与费用报告](qwen_plus_prompt_cost_v6_20261104.md)读取144次历史API回执的图像、文本、思考和正文细分。v4真实结果见[12任务报告](r2r_v4_batch12.md)；新prompt估算不冒充新模型实测。

## 固定设置

| 项目 | 本阶段配置 |
|---|---|
| 实际 API 模型 ID | `qwen3.7-plus` |
| 服务 | 北京 DashScope，`https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 接口 | Chat Completions，`response_format: {"type":"json_object"}` |
| 思考 | `enable_thinking: true`，`thinking_budget: 2048` |
| 输出预算 | `max_output_tokens: 8192`，请求中映射为 `max_tokens`；不是要求答案写满 8192 tokens |
| 超时 | 每请求 180 秒 |
| 采样参数 | 不发送 temperature、top_p、seed，使用供应商默认值；不承诺可重复生成完全相同文本 |
| 教师模板 | `v07-teacher-user-supplied-v6`；完整原文和 SHA-256 在配置中 |
| 证据模板 | `plain-evidence-v2` |
| 输出协议 | `current-categories-v1` + `compact-v2`，英文 `landmark / local / route` |
| 图像 | RGB 原分辨率 JPEG quality 95；地图与检测叠图保持原 PNG；不额外改变当前预处理 |
| 失败策略 | `skip_invalid_keep_last_valid_v1` |
| 审核 | 原文、失败帧、上下文断点全部保留；格式通过仍不等于人工批准 |

JSON object 模式保证的范围不等同于本地字段校验，**并未把 Local 的 maxLength 约束下推给服务端**。Local 90 字符、Route 600 字符、最多三类等既有输出校验继续保留；90字符上限允许81至90字符的回答进入审核；超过90仍拒绝。单帧失败继续后续任务，不自动修写原文。

模型名称是供应商别名，不能据此宣称远端权重永久冻结。归档实际返回的模型字段和可获得的版本信息；如供应商行为变化，需重新比较。本地固定配置、提示原文和哈希。构建请求时检查提示哈希；配置整体哈希由锁文件及回归测试核验。修改模型、提示、图像处理或预算时另存新配置版本，不能覆盖旧实验的运行配置。

## 输入、输出和递归条件

继续使用原始 instruction、最近一次有效 Route、有效期内的历史 Local 与相对运动、当前及历史 RGB、当前检测叠图、此前已提交的语义地图。典型人工 demo 请求有六张图：三张历史 RGB、当前 RGB、地图、检测叠图；历史不足时按实际有效输入归档。

教师只输出类别数组、位置关系与剩余指令；不生成 image_id、current_subgoal、status、waypoint 或参考线。来源 ID 外部绑定，waypoint/ref_line 仍由轨迹监督产生。Landmark 必须有当前检测及 RGB 支持，地图中的历史地标可辅助 Local，但不能冒充本帧新检测。保留当前类别准入、投影、OCR 和独立地图规则。

每个计划关键帧最多一次语义请求，不自动补问、裁剪、润色或换模型修复。改变后的处理如下：

| 情况 | 当前帧 | 后续帧 |
|---|---|---|
| 超长一个字符、非法字符、非法 JSON、截断、无依据 DONE、非法输出来源 | 保存响应和失败原因，禁止提交地图及工作 Route/Local | 继续，用最近一次通过校验的上下文 |
| 第一帧无有效回答 | 保存失败，地图不接受该帧语义结果 | 使用原始 instruction；无有效历史 Local |
| API/网络异常、预算或重复请求保护拒绝 | 记录错误，不自动重试 | 停止，后续明确标为未执行 |
| 输入来源/哈希不符、地图写入或文件写入异常 | 不当作普通模型坏答案 | 停止，先解决数据或存储问题 |

上一帧失败后，地图的**语义层**沿用最近提交版本，几何层仍可包含截至当前请求之前的真实深度观测。Local 的相对运动从最近有效 Local 的源位姿计算，不假定来自紧邻帧；超过现有 **30 秒**记忆时限则失效，输入按 unknown 处理。原始 Local、来源及过期原因仍归档。不得用人工答案填补空缺。

每帧 `context_recovery` 记录上下文来源、跳过帧 ID、首轮回退及没有人工回填；失败行记录 `failure_stage / failure_action / label_eligible`。闭环报告分别给出实际请求数、`all_frames_attempted`、`all_frames_completed`、`fatal_stop`。例如 10 帧中一帧超长，可以得到 10 次请求、9 帧有效，但仍是 `all_frames_completed=false`，不能报告 10 帧标签全部通过。

旧策略可显式设置 `annotation_failure_policy: stop_on_invalid_v1` 用于历史对照。新建标注运行默认使用跳过策略；旧报告不重写。

## 成本与运行

保留原实测配置中的人民币估价：每百万输入 tokens ¥2、输出 tokens ¥8（2026-11-04复核的基础单价，不计限时优惠，估算而非账单）。共享预算账本仍有 100 美元上限；现有账本采用 **1 元按 1 美元占用预算**的保守规则，这不是汇率换算，实际人民币估价单列。思考用量按供应商返回的 usage 记录，不凭空推算服务端分阶段耗时。扩大批次前须依据实际账单与当前价格核验预算。

在仓库根目录对**新的已准备数据**执行：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.semantic_loop run \
  --prepared /path/to/new/prepared \
  --config configs/dataset_preparation/qwen_plus_teacher.v6.json \
  --credentials /home/xinglinlu/.config/llm/credentials.env \
  --output /data/ticvla/outputs/stage1a/plus-next-001

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.semantic_loop audit \
  --output /data/ticvla/outputs/stage1a/plus-next-001
```

`prepared` 使用已有 prepare 流程生成，不是原始 episode 目录；输出必须为新目录。配置只存 `DASHSCOPE_API_KEY` 环境变量名，密钥不入仓库。预算账本是当前机器上的既有绝对路径，换机器时显式创建新配置版本并迁移预算记录。

相同请求受账本去重保护。不能直接重跑旧的 10 帧并通过删除账本绕过保护；需要重测时单独定义重测批次与预算。此次单元验证使用模拟 API 响应，验证超长帧后的继续执行和地图隔离，不计为新的 Plus 模型质量实测。

交付审核页继续遵守 [Agent.md](../Agent.md)，展示全部关键帧、模型原文、地图、上下文跳帧和过期信息。下一阶段仍须人工核验 Local 的编号与方向、Route 进度、误检入图及错误传播，才可批准训练标签。

## 本次工程验证

68 项相关测试通过：`test_semantic*.py` 52 项；失败策略、Local 上下文、紧凑 Route 与 Oracle 执行测试共 16 项。覆盖第二帧 Local 超长一个字符后第三帧仍执行、坏帧不提交地图、不传播被拒绝的 Route、无依据 DONE 不传播、旧停止策略兼容、API/预算异常停止及冻结配置哈希检查。调用与地图写入依赖在调度测试中使用隔离替身；这验证工程行为，不表示 Plus 已完成新一轮 10 帧真实评测。

## Local 90 与明确完成规则更新

Local 必须简洁且不得超过90字符（空格、标点计入），不是90 tokens。不自动裁剪。教师 v2 与部署生成语法同步；81至90字符可通过，91及以上拒绝。

字符串使用英文文字、数字、空格和允许的普通标点；`~` 写成 `about`，角度符号写成 `degrees`，箭头写成动作词。保留编号中的 `/`。JSON 本身的括号、双引号等仍遵循 JSON 语法。

新增简短输入 `Completion evidence: confirmed|unverified`，由当前样本已有三类事件共同决定，不补造到达证据。只有 confirmed 且整项任务确实完成时，Route 才能恰好为大写 `DONE`，无前后文字。其余情况输出剩余/等待指令，不使用 DONE。此标志只使已有完成门槛对教师可见，不提供未来轨迹或隐藏目标。

本次不重新调用 API。历史回答只做独立规则复核；原地图、递归上下文和人工批准状态不变。规则通过不等于语义正确，也不意味着新 prompt 的实测效果已经验证。

本次独立复核：40帧中旧严格规则通过25帧，新90字符规则通过30帧；剩余6帧超过90、1帧含不支持的字符、3帧不规范且无证据的DONE。未重放递归链，新增通过的5帧不自动入图或批准。审核见运行目录 `local90-revalidation/index.html`。55项语义回归和32项协议/审计/来源相关测试通过，共87项。

## 2026-10-30：v4 观测依据规则

当时新建标注默认v4，替代未实测的v3；2026-11-04起默认v5。物体、地点、空间关系只需当前Local、图像或地图中至少一项支持；不是必须出现在Local文本中。Previous Route仍合理时可以原样保留。未观测任务目标仍可作为寻找目标，但不能虚构其位置或方向。原指令中的目标与未完成要求不因暂时不可见而删除。

模型、思考2048、输出8192及Local90保持不变。v1/v2/v3与历史实验保持原样；此前6/8结果属于v2。此次未调用API，v4实际效果待验证。

## 2026-11-04：v5职责分离与精简

新建R2R试点默认v5；其他通用入口显式传入v5配置。先定义Landmark为当前支持的类别、Local为当前位置与朝向、Route为剩余导航命令，再集中说明字符、完成与坐标要求。Route不复制Local、不累计已完成动作，也不能将等待命令写成已在等待的事实。保持90/600字符、最多三类、完成标志、2048思考及8192输出上限。未恢复第二次推理，未自动改写旧标签。v5质量及输出耗时仍待真实对照。

## 用户指定v6

默认R2R试点改用v6，其他入口显式传入配置。保留原文字面内容，包括`Rejecthistory-only`、两处句号后无空格，以及标点限制与`Inferred:`不一致。解析器仍为现有兼容范围，不新增符号拒绝规则。v5为前一轮方案，不再是新任务默认配置。此次无付费API请求。
