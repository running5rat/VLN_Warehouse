# 完整Route与连续曲线跟踪 Demo v2

使用人工标注 `manual-static-004` 的原始静态仓库。两段任务在同一Isaac Sim 5.0 episode、同一地图与odom中完成，车辆由真实轮速/物理仿真驱动。

本轮按约定由Codex编写的决策代理代替VLA，实际provider为`codex-reviewed-decision-replay`。真实运行YOLO-E、PP-OCR、深度投影、语义关联、PCHIP拟合与Pure Pursuit；没有真实LLM、Reference Decoder或FMAE推理。旧runtime报告中的`provider=codex-session`是固定字段，整理报告已按七份实际响应更正，并保留原字段供审计。

## 两个完整任务及剩余Route

任务一：沿起始通道到远端，右转横穿横通道，再右转进入第二条平行通道，走到近端，右转回起点，完成环绕建图并STOP。

首次实际Route：

> Follow this aisle to the far end, turn right across the cross-aisle, turn right into the second aisle, follow it to the near end, turn right back to the starting point and STOP.

后续依次保留“横穿远端横通道→右转→第二通道返回→近端右转→回起点→STOP”和“到近端→右转→回起点→STOP”。完成并实际停车后才输出STOP。

任务二：前往本轮地图中更远第二通道的`04/5R`旁STOP。没有沿用旧Demo的03/4R，也没有使用旧地图目标坐标。本轮查询时04/5R为identity_confirmed，源帧相机视锥外，地图位置约`(15.27, -6.65, 2.59)m`。

首次实际Route：

> Turn around, cross the near-end cross-aisle, turn left into the second aisle, follow it to sign 04/5R, approach the sign and STOP beside it.

进入第二通道后，更新为：

> Continue along the second aisle to sign 04/5R, approach it and STOP beside it.

七次响应全部接受，没有重试或人工修改旧输出。每次请求的`input.json`保存上一份已接受Route；第二任务重新从自己的完整初始Route开始。七次递归检查和终末停车指令检查均通过。`STOP beside it`也是末尾停车命令，不应仅因STOP不是最后一个单词而判失败。

完整输入、源RGB、冻结地图、原文和曲线数组见总览的七个决策卡片；字段语义仍待人工审核，`human-review.json`没有自动批准训练标签。

## 运动与提前接管

没有提高车辆最高速度，仍为0.3m/s。环绕由三段五点曲线覆盖，第二任务由两段覆盖；在旧曲线剩余约1.8m时提交下一段。决策进程工作时控制进程继续跟踪旧曲线，合法新曲线按当前odom裁剪接管。

| 指标 | 实测 |
|---|---:|
| 环绕路程 | 51.224m |
| 回到起点误差 | 0.075m |
| 两段任务总路程（地图源帧累计） | 70.470m |
| 最终距目标可见表面中心的平面距离 | 1.784m |
| 最后1秒平移速度 / 转动速度 | 0.00111m/s / 0.000082rad/s |
| 第一次曲线接管线速度 | 0.30115m/s |
| 第二次曲线接管线速度 | 0.30098m/s |
| 第三次曲线接管线速度 | 0.30124m/s |

三次接管时均在tracking，运动未因等下一段waypoint而中断。命令样本7891条；9条sensor_expired均发生在启动观察阶段、首个导航响应之前，command_expired为0。排除初始化观察和等待提案后，低于0.02m/s且低于0.02rad/s的静止样本约1.27%，包含两个任务正常停车；不是丢弃这些停车后计算的指标。

`motion-continuity.png`显示实测速度及切换时刻。由于本次runtime启动后才增加accepted_monotonic日志字段，切换时刻采用响应接受的wall时间，通过本机wall/monotonic时钟偏移换算，与最近控制样本相差2–24ms，换算口径保存在`continuity.json`。

这验证了独立跟踪与提前切换的工程链路；代理是Codex编写的代答逻辑，不能据此声称某个真实LLM已满足推理延迟。以后接入真实LLM时，必须在旧曲线覆盖耗尽前返回合法结果，否则仍应停车。

## 建图与未通过项

采集1317帧，1315帧在任务结束前进入地图；关闭仿真期间额外采集的2帧未补算在线完成。实际处理255个分割/OCR结果，最后保留37个候选地标，全部仍为tentative，静态锚点为0。不能把候选数当作正确实例数。

分割p50/p95约16.6/48.1ms，OCR约425/863ms。感知源模拟时间到消费者的差值p50/p95约0.4/1.0s；没有把它当作浏览器端到端延迟。

**OCR冲突仍存在。** 环绕及第二任务接近时多次正确读到04/5R，但最后接近后标牌被画面上沿裁切；末次读数为15R，分块合并也保留了04/5R与04 | /5R的格式差异，最终身份状态为conflict。任务保留开始时绑定的UID，未用目标指令覆盖OCR。导航驶近停车已完成，持续文字身份核验未通过。

`target-approach-rgb.png`显示接近时完整可读的04/5R；`target-final-rgb.png`是最后真实视图，编号已部分出画，不能声称最终RGB仍完整显示04/5R。对应源帧524、1201、1302的mask、文字与绑定记录保存在`perception-evidence/`。

## 录像与交付

视频不再叠加背景介绍、模型设置，只显示RGB、Semantic map、Landmark、Local、Route、Waypoint、完整Instruction及输出来源。Local等属于最近一次输出的源帧，不能解释为每个视频帧重新推理。

- `live_demo.mp4`：5fps完整连续采样，1323帧、约264.6s；墙钟采样跨度约264.45s，主要看此原速版本判断运动。
- `demo_4x.mp4`：明确加速的概览，不用于证明原速实时性。
- `continuity.json`、`decisions/`：递归Route、接管实测及全部请求响应。
- `trajectory-and-landmarks.png`、`semantic-3d.png`、`live-final-map.png`：轨迹、候选几何与最终地图。
- `map-local/`：完整本地SQLite与几何数组的链接；发布或搬机器时需另携带原目录。

原始目录为`/data/ticvla/outputs/navigation/codex-continuous-20261112-v2`。运行没有覆盖旧Demo，没有调用付费API。为避免资源争用而暂停的R2R数据准备已自动恢复。

最终检查：Demo接口7项、实时感知21项、导航13项，共41项通过；页面链接、HTTP视频、SQLite完整性，以及七份递归Route与三次运动中接管检查通过。详见tests.json。
