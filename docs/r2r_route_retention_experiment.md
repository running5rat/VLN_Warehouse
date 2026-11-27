# 仅递归 Route：未完成步骤保留实验

本配置用于用户指定的 r2r-10148 十帧重标注，不更改既有批次和历史结果。

配置：[qwen_plus_teacher.route_only_preserve.v1.json](../configs/dataset_preparation/qwen_plus_teacher.route_only_preserve.v1.json)。

`task_input_mode=route_only`：不单独发送 Instruction；第一帧 Previous Route 为原始完整指令，后续采用最近合法的模型 Route。原任务仍保存在审计侧，不暗中输入后续请求。

只在 SYSTEM 的 `route:` 段增加以下要求；Landmark、Local 和其长度等规则均保持不变：

> Preserve every unfinished step of Previous Route, in order, including later turns, targets and the final stop or wait requirement. Do not truncate, omit or replace unfinished steps with only the immediate next action. Remove a step only when current observations and relative motion support that it has been completed; if uncertain, retain it. Shorten wording rather than dropping unfinished steps to meet the length limit.

Local 保持描述当前定位，不能把上述任务步骤保留要求施加给 Local。未增加 Stop 锁定、语义自动修复、第二次提问或隐藏任务补偿。Route 最多600字符，Local最多90字符。非法输出隔离当前帧，继续使用最近有效上下文；第一帧非法时，下一帧仍从原始初始 Route 开始，不能称为采用了被拒绝答案。

模型为冻结的 Qwen3.7-Plus 配置。固定源帧为0、2、12、24、30、43、53、61、64、65；复用分割/OCR、独立重建递归语义地图与 Local。只有提示规则变化，后续自递归上下文可以分化，不宣称是逐帧同输入盲测。教师不生成 waypoint/ref_line，这两项由示范轨迹提供，仅用于审核。

产物：`/data/ticvla/outputs/stage1a/r2r-10148-route-preserve-20261105`。每帧一次真实请求，不自动重试。复查已有结果请打开产物的 `comparison.html`、`COMPARISON.md` 和 `episodes/r2r-10148/review/index.html`，不要重复启动已完成的付费运行脚本。

## 本次实测结果

10/10响应、8/10规则有效。第61、64、65帧连续输出`stop.`，未再引入浴缸目标。首帧仍截断未完成步骤，第43帧省略进门几步；两帧同时因Local超长被隔离。第2帧通过既有“尚无有效输出时使用初始Route”策略恢复，不能宣称新增规则彻底解决了任务丢失。费用估算¥0.115564，API累计566.90秒。完整报告为上述产物目录的`COMPARISON.md`，全部结果待人工审核。
