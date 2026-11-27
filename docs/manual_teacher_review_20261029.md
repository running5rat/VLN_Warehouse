# 人工场景：当前标注规范三模型对照

2026-10-29，使用 `manual-static-004` 的全部 10 个预定关键帧。教师采用 `docs/vlm_annotation_and_io_spec.md` 3.2 的详细模板，版本 `v07-teacher-20261029`，证据格式 `plain-evidence-v2`。没有修改车端默认 SYSTEM。

## 查看结果

- [浏览器完整审核](http://localhost:8789/review/)
- [详细报告](/data/ticvla/outputs/stage1a/manual-review-20261029/review/REPORT.md)
- [整段任务对照视频](/data/ticvla/outputs/stage1a/manual-review-20261029/review/episode.mp4)
- [人工审核模板](/data/ticvla/outputs/stage1a/manual-review-20261029/review/human-review.json)
- [产物核验](/data/ticvla/outputs/stage1a/manual-review-20261029/review/artifact-audit.json)

这些是本机文件，不是 GitHub 托管页面。服务器停止后，从运行机器重新启动：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8789 \
  --bind 0.0.0.0 \
  --directory /data/ticvla/outputs/stage1a/manual-review-20261029
```

远程浏览器需转发该端口；HTTP 代理应绕过 localhost。

## 实测结果

| 配置 | 实际返回/预定 | 协议有效 | 后续未运行 | API p50/p95 秒 | 费用估算 CNY |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-Max，思考预算 8192 | 5/10 | 4 | 5 | 33.56/89.66 | 0.680256 |
| Qwen3.7-Plus，思考预算 2048 | 4/10 | 3 | 6 | 27.99/34.11 | 0.089162 |
| Qwen3.8-Omni-Flash，非思考 | 1/10 | 0 | 9 | 4.17/4.17 | 0.003495 |

共 10 次真实新调用，估算 ¥0.772913，包括失败响应，使用实际 usage 和[北京端点未折扣牌价](https://help.aliyun.com/zh/model-studio/model-pricing)，不扣缓存/促销优惠，非实际账单。保留原共享预算上限；GPT pending，GLM 本轮未调用。

**三种配置都没有完成整段递归标注，不能作为无人审核批量 GT 的通过结果。** API 往返时间包含网络和服务端处理，不是纯 decode。Omni 只有一个响应，不具备统计稳定性。不同模型后续上下文不同，不能把所有时刻当作同输入比较。

具体问题：

- 前三帧 Max、Plus 均能利用可读标牌表达 Local，并保留“直行、尽头左转、到灭火器”的任务顺序。仍登记普通重复货架，独特性选择不足。
- 第 1280 帧，检测器把左侧立柱黄色条形标识误报为灭火器；Max、Plus 均采纳该建议。Max 的该轮协议有效，因此误报进入地图；第 1365 帧又被引用为左侧约 2.4 m 的灭火器。位置来自地图，不代表类别或目标身份正确。
- Plus 第 1280 帧 Local 长 81 字符，Omni 首帧长 95 字符，均超过 80。Plus 因整轮拒绝，没有将该帧误报写入自身地图。
- Max 第 1365 帧使用 `~`，不符合当前 Route 字符语法；虽然它属于 ASCII，仍被拒绝。原错误消息比较笼统，具体原因由原文和同一解析器定位。
- 供应商 JSON 模式没有约束字段长度或字符集合。当前模板未穷举部署允许字符；这是提示/解码/校验需对齐的问题，不能把此类中断全部解释为模型不理解导航。

建议先对齐生成约束与部署解析器，并为 Local 留长度余量；再做小目标原图裁剪和检测误报核验对照。不能通过静默裁剪、纠字或放宽解析，把错误地标变为正确标签。Plus 可保留为低成本待审核候选，本轮没有证据证明 Max 在此任务上值得更高成本，也没有足够结果选择 Omni 为完整教师。

## 实验与审核边界

三个模型复用同一真实 YOLO-E/TensorRT、CPU OCR 缓存，但各有独立地图、Route、Local。首次输入哈希一致；后续由自身合法响应递归更新。输入地图只含已提交的历史语义，当前几何可用；没有人工历史修复。格式失败即停止，全部 30 个模型×帧槽位均保留，20 个未运行位置不补造输出。

模型输入包含三张历史 RGB、当前 RGB、先前语义地图和检测框叠图，不发送事件表、文字地标表、未来轨迹或本轮人工答案。像素 mask 与执行事件仅供审核。教师只生成 Landmark、Local、Route；waypoint/ref_line 不适用，不运行 Decoder 或车辆控制。

视频为实际录制的连续 RGB 回放，152.6 秒，按源关键帧更新回答；不能据此宣称回答在在线源时刻已到达。完整输入与传输图像、原文、usage、投影/OCR、地图、失败原因、人工导出控件均在审核页。Agent 的视觉/日志判断不是用户批准，人工批准数为 0。

相关接口测试 53 项通过。产物核验检查 209 份来源文件哈希、250 个网页链接、30 个预定位置和完整视频解码/源时间线；无坏链。此检查仅证明归档和展示完整，不证明标签语义全正确。

冻结配置、模板、代码快照和执行脚本位于 `/data/ticvla/outputs/stage1a/manual-review-20261029/`。重建报告可使用 `semantic_loop_review build/video/audit`，不需 API；付费请求受预算账本去重保护，不能删除账本后自动重发。
