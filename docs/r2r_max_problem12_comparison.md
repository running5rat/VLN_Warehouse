# Qwen3.8-Max 问题任务对照

对上轮Plus暴露的问题任务进行定向诊断，不是随机评测集，也不用于宣称整体模型准确率。固定12任务139个原始关键帧，完整任务递归重标注，未追加新的训练数据或人工答案。问题检查清单保存为产物`review-checklist.json`，不输入模型。

配置以当前Route保留提示为准：仅route段要求保留全部未完成步骤，不向Local/Landmark增加此约束。不单独输入Instruction，首帧Previous Route初始化完整原任务，后续用最近合法Route；Local、地图独立递归。模型改为qwen3.8-max，思考预算仍2048、输出上限8192、超时180秒，与Plus对照一致。无自动补问、修复或重试。API/预算异常停止，不静默换模型。

冻结使用同一RGB、历史图像、YOLO与OCR缓存、odom及原始任务，两个模型各自建立地图和历史上下文。后续实际输入会分化；报告属于闭环策略对照，不能描述为每帧完全相同上下文下的单轮模型比较。先前Plus答案不发送Max。

官方北京价格于2026-11-06核对为输入12元/百万、输出36元/百万tokens：[价格表](https://help.aliyun.com/zh/model-studio/model-pricing)。成本含思考输出，不双计，按实际usage估算并保留冻结价格，非供应商账单；预计本批约8–15元。沿用共享预算和账本，不扩大上限。

产物：`/data/ticvla/outputs/stage1a/r2r-max-problem12-20261106`。

- [实时统计](http://localhost:8797/statistics.html)
- [完整审核与视频](http://localhost:8797/)
- [Plus/Max逐帧对照](http://localhost:8797/comparison.html)（标注和审核生成后）

报告分别统计全部139帧的返回、规则有效、全任务规则通过、来源契约、费用耗时，以及上述具体错误是否消失/持续/新增/无法判断。规则通过不自动标为语义通过，用户人工批准仍为0。图像和示范waypoint/ref_line分别展示；后者不输入教师，也不是Max生成的几何输出。
