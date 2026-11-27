# Qwen 标注模型分级对照

最新对照：[2026-10-29 人工场景、v0.7 教师模板与语义地图递归](manual_teacher_review_20261029.md)。三配置均未完成全部 10 帧，具体格式失败与错误地标传播分别报告；下文保留 10 月 27 日历史结果，不混用输入与成绩。

2026-10-27：GPT 标注实验按用户要求保持 pending。本次仅比较 Qwen，未改变生产教师模型，未批准新训练标签。

本机结果：

- [浏览器并排审核页](/data/ticvla/outputs/stage1a/qwen-tiers-20261027/index.html)
- [实测结论与失败分析](/data/ticvla/outputs/stage1a/qwen-tiers-20261027/REPORT.md)
- [实际费用及训练集规模外推](/data/ticvla/outputs/stage1a/qwen-tiers-20261027/cost-projections.json)

这些 `/data` 路径是本机文件，不是 GitHub 托管内容。可启动静态服务器：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8788 \
  --bind 0.0.0.0 \
  --directory /data/ticvla/outputs/stage1a/qwen-tiers-20261027
```

在运行机器访问 `http://localhost:8788/`；远程机器用宿主机地址，容器需要映射端口。

## 本轮结论

准确性优先时仍保留 Qwen3.8-Max 思考模式。低成本的 Qwen3.7-Plus 有限思考（2,048 token budget）适合继续做需人工审核的提议对照，尚未证明与 Max 等效。Omni-Flash 的纯 API 成本最低，但长指令递归删除步骤等问题阻止其直接成为唯一教师。

10 人工关键帧和 8 个 R2R/RxR episode 的 32 个时刻，共42位置/配置。五配置中复用旧Max思考，其余分别为Max非思考、Plus非思考、Plus有限思考、Omni非思考。10个人工帧和8个递归首帧的因果输入完全一致；后续递归输入各自更新。另有1个超视距地图案例，不混入主实验成本和评分。

新增157次Qwen调用、估算¥3.53194；复用41条旧响应。未启动GPT调用。费用包含失败响应，未包含人审，未扣缓存/促销/Batch优惠，不是账单。

18个共同输入的Codex辅助复核中，同时符合协议/规则且未发现明确语义问题的数量：Max思考16，Max非思考11，Plus非思考11，Plus有限思考12，Omni非思考10。它们不是独立盲测准确率，未自动作为批准标签导出。

## 复现接口

使用 `ticvla.warehouse.dataset_tools.qwen_tier_review`，复用现有 `run_pilot`、同一协议和持久预算账本。凭据从外部文件作为数据读取，不写入仓库或请求报告。

```bash
# 新实验：只准备冻结输入和配置，不提交 API 请求。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.qwen_tier_review prepare \
  --source /data/ticvla/outputs/stage1a/oracle-review-20261026 \
  --output /data/ticvla/outputs/stage1a/qwen-tiers-NEW

# 提交小批真实请求。旧Max响应直接复用；其余配置沿各episode顺序生成。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.qwen_tier_review run \
  --output /data/ticvla/outputs/stage1a/qwen-tiers-NEW \
  --credentials /home/xinglinlu/.config/llm/credentials.env

# 单独复测已冻结的超视距场景，不运行Isaac，不发送车辆命令。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.qwen_tier_review map-probe \
  --output /data/ticvla/outputs/stage1a/qwen-tiers-NEW \
  --source /data/ticvla/outputs/map-memory-review-20261026/query

# 只更新静态审核页和统计；不调用模型。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.qwen_tier_review summarize \
  --output /data/ticvla/outputs/stage1a/qwen-tiers-20261027
```

已完成的相同目录可通过 `run` 恢复未开始的bundle，不重复已完成请求。中断时留下的进行中请求需先检查状态；预算账本禁止相同付费请求自动重发，因此换一个新目录也不会绕过去重。需要重新采样时必须建立明确的新实验配置和记录，不能删除账本重试。

原文、解析失败、提前完成、递归中断均保留。`assistant-review.json` 是会话辅助评估；`human-review-template.json` 是尚未批准的人工审核模板。不会因格式通过或辅助评估通过自动将标签导入训练。
