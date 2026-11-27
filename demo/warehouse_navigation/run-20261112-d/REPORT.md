# 原仓库环绕建图与视野外标牌导航

本次使用人工标注 `manual-static-004` 的原始仓库、原始编号、Nova Carter 和相机起点，不使用改变编号的 `original` 外观变体，不添加动态人物。Isaac Sim 实际版本为 `5.0.0-rc.45+release.23960.184afb15.gl`。

两个任务在同一物理仿真、同一 odom 和地图中执行。**环绕和目标驶近停车均已发生；持续编号识别仍存在冲突，训练 VLA 的能力没有验收。**

## 实际结果

| 项目 | 实测结果与口径 |
|---|---|
| 任务一 | Complete one circuit of the warehouse aisles and build a semantic map. |
| 环绕 | 52.56 m，起终点平面误差0.198 m；实际轮速驱动，没有初始化后的位姿瞬移 |
| 任务二 | Navigate to shelf sign 03/4R and stop beside it. |
| 目标来源 | 本次真实分割、OCR、深度投影所得地图；任务发出时目标在相机视锥外 |
| 目标绑定 | `f5febf048909438fa28ac429d8c6514e`；任务开始时读数03/4R，身份为candidate |
| 到达 | 距地图中的可见表面中心1.757 m（平面距离）；最终RGB可直接看见03/4R |
| 停车 | 前一秒位移测得0.00103 m/s；末条Isaac执行反馈v≈0.00096 m/s、w≈0.00027 rad/s，命令为0 |
| 总路程 | 地图源帧累计63.43 m；Isaac完整结束报告63.47 m，采样/结束时间不同 |
| 采集 | 5865帧RGB-D；5860帧在任务结束前进入地图。最后5帧为任务结束与仿真关闭之间的尾帧，未补算在线建图 |
| 语义处理 | 跨运行时重启合计890个真实分割/OCR结果；原runtime报告708只计重启后，不能作为总量 |
| 最终候选 | 33个地图记录：27个sign、6个fire extinguisher；不等于33个正确独立目标 |
| 最终静态锚点 | 0；所有物体仍为tentative，不因可读编号而授予静态资格 |
| 决策 | 共10次：8次接受、2次因源帧超过30模拟秒拒绝；全部输入输出均保留 |

没有用仿真实例真值量测IoU、绝对定位误差或碰撞率，不报告这些指标。到达阈值是本Demo约定的2m平面邻域，不是R2R等benchmark标准。

## VLA替代与控制来源

本轮按用户要求由**Codex会话代答** Landmark、Local、Route和五个源车体系waypoint。实际控制为：

`五个waypoint → PCHIP曲线 → 根据当前odom裁取剩余段 → Pure Pursuit → 差速轮控制 → Isaac PhysX`

曲线数组保存为每个决策的 `curve.npz`。没有调用InternVL、Reference Decoder、KV压缩或FMAE；`raw.txt`中的REFLINE仅保留接口标记，不能解释为Decoder已前向。导航API调用为0。

本次沿用消费者的旧事件字段`source=llm`，它不能作为实际调用模型的证据；帧级selection中真实provider是`codex-authored-category-policy`。交付代码已为这种固定策略改为显式`event_source=perception`，历史数据库不回写。

Codex参考了既有仓库环绕结构，逐次提交源帧决策；本次不是未知布局泛化测试。任务二的位置从本轮地图查询，没有读取目标世界真值。真实分割器为YOLOE-v8s固定词表TensorRT FP16，OCR为PP-OCRv5 mobile。建图准入由Codex固定指定sign、label、fire extinguisher、door，是类别策略，不是每帧真实VLM选择。

全部10次决策见总览和 `decisions.jsonl`。早期请求未冻结地图PNG，输入JSON保存了当时地标表；不能用最终地图冒充早期视觉输入。后期任务请求已冻结输入地图。`visual_context_source_image_id`区分人工实际查看的RGB与提交时更新的odom源帧。

## 明确保留的错误与限制

1. **返回途中OCR冲突。** 目标最初及返回途中六次读数归一为03/4R；源帧5679将文字读为`03 | 14R`。它与03/4R无子字符串包含关系，按已有规则不再登记唯一文字，最终为`identity_status=conflict`。任务仍保留开始时绑定的UID，未用用户指令覆盖OCR。实物驶近成功，不代表最终文字身份持续核验通过。
2. 环绕结束时地图包含六个完整编号：03/4R、04/4R及01–04/5R。最终地图因新读数冲突仅保留四个5R完整编号。所有历史证据仍在数据库；轨迹图红圈03/4R明确是任务开始时的绑定，不是最终无冲突文字。
3. 地图还有BOX、FRAGILE等低价值文字及可能重复/误关联记录。33个记录没有逐项人工批准，不能据此声称地标精度达标。
4. 源帧过期导致loop-02、loop-05拒绝，之后各提交了一次新源帧请求。没有修改被拒绝的旧原文。
5. 初期串行地图/OCR和绘图阻塞控制，触发传感器/命令超时停车。本episode中保持Isaac运行，重启地图runtime并将控制分为独立进程，随后继续完成任务。完整执行日志含4564条sensor_expired和2074条command_expired，不能把本次说成全程稳定实时。
6. 交付代码补充了“拒绝的新提案清空旧曲线”和“完成前必须观察停车”的守卫；5项接口测试通过。本次实际完成前也独立检查了停车，但这些补充代码没有另跑第二个完整episode。
7. 之前启动/调试尝试保留在同级无后缀、`-b`、`-c`目录；本报告只评估连续物理episode `-d`，未拼接不同车辆运动冒充一圈。

## 耗时

890份已返回结果：分割p50/p95为17.3/46.6ms；OCR为199/965ms。源模拟时间到地图消费者的差值p50/p95为0.4/1.4s。最后一项不是浏览器绘制延迟，也不是完整墙钟端到端延迟。控制独立进程记录的源年龄p50/p95约0.232/0.769s。

几何采集目标5Hz、语义采样目标1Hz，控制有独立超时保护。本次没有通过此前约定的全链路实时验收；不能以模型单次推理较快替代实际完成频率。

## 视频、地图与审核

- `live_demo.mp4`：整个episode发布面板的连续采样录像，保留调试等待；2fps编码，2261帧、1130.5s。实际采样跨度约1174s，部分采样耗时超过0.5s，因此不是精确墙钟等速；`video-timeline.jsonl`保存每帧真实时刻。
- `demo_4x.mp4`：上述录像4倍播放，不删除失败阶段。
- `target-start-rgb.png` / `target-final-rgb.png`：目标视野外任务开始图和驶近停车图。
- `trajectory-and-landmarks.png`、`live-final-map.png`、`semantic-3d.png`：真实轨迹、最终地图和投影实例表面。
- `perception-evidence/00001438`与`00005679`：最初正确识别和最后冲突的RGB、mask、文字裁剪及绑定记录。
- `map-local/`：指向完整本地SQLite＋不可变几何数组，约12GB；必须连同数组使用。它是本机链接，不是可携带的Git文件。
- `human-review.json`：待人工审核，未自动批准为训练标签。

完整原始采集、地图、所有感知帧和控制日志：`/data/ticvla/outputs/navigation/codex-map-return-20261112-d`。Isaac原始capture-report中的`loop_complete=false`对应旧固定waypoint runner没有运行；本Demo的两任务控制使用navigation入口，环绕证据独立保存在`loop-complete.json`。

复现命令见上级README与`decision-plan.json`。自动回放明确标记为已审核Codex决策回放，不是新的在线语言推理；目标必须从新一轮真实地图重新检出，缺失或歧义时失败。后台R2R仅数据准备任务已恢复，没有启用付费标注。

最终检查：Demo接口5项、实时感知回归21项、导航回归13项，共39项通过。脚本语法、入口help、地图SQLite quick_check、审核页本地链接通过；审核页与视频HTTP均返回200。检查结果另存`tests.json`。没有再运行第二个完整物理episode来替代本次记录。
