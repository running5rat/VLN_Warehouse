# 环绕建图与视野外标牌导航

2026-11-12 新约定：[RGB 像素 waypoint、同帧深度投影及 Oracle 对照](../../docs/rgb_pixel_waypoint_demo.md)。以下 v1/v2 实跑使用旧的车体系米制 waypoint，不能作为像素协议的闭环成功证据。新对照报告保存在 `/data/ticvla/outputs/navigation/rgb-oracle-20261112-v1`，每个模型覆盖环绕建图与前往第二通道04/5R两个任务；审核服务使用8803端口。

场景使用 `manual-static-004` 的原始仓库、原始编号标牌、相机和 Nova Carter 起点，没有动态人物。

已完成实跑：[视频与逐次决策](run-20261112-d/index.html)、[验收说明](run-20261112-d/REPORT.md)、[轨迹与地标](run-20261112-d/trajectory-and-landmarks.png)。绕环52.56m、闭合误差0.20m；随后从本轮地图选择视野外的03/4R，实际驶近并停车，距离地图表面中心1.76m。最终OCR出现03/4R与03|14R冲突，未改写成无冲突结果。

本 Demo 将 VLA 的接口暂时交给 Codex 会话：Codex 查看当前 RGB、已观测地图和源位姿，提交 Landmark / Local / Route / 五个 waypoint。运行日志明确标记 `codex-session`，不声称经过 InternVL、Reference Decoder 或 Flow 前向。将来接入真实 VLA 时替换提案来源。

车辆实际执行链路为源帧 waypoint → 按弧长参数拟合 PCHIP 曲线 → 随 odom 更新的剩余曲线 → Pure Pursuit → 差速轮速 → Isaac PhysX。0.3m/s上限；命令、传感器过期以及已占用/未知空间检查仍保留。没有合法提案时停车。

感知使用真实固定词表 YOLOE-v8s TensorRT 和 PP-OCRv5 mobile。Codex 为本演示指定登记 sign、label、fire extinguisher、door；这是明确的类别策略，不是逐帧 InternVL 答案。所有深度进入几何观测，mask 与文字只绑定其源帧实例，新地标仍为 tentative。

两个任务保留同一 odom 与地图：

1. 环绕仓库通道并回到起点。记录实际行进距离、起终点闭合误差和全部控制命令。
2. 从本轮地图查询当前 RGB 视锥外的编号标牌；记录任务发布时的投影检查、身份、位置与证据。到达目标表面中心的平面距离2m内停止。不得读取标牌目录或仿真目标真值补足感知。

## 启动与观察

### 连续曲线与完整Route版本（v2）

**已完成新实跑：**[原速视频与全部输出](run-20261112-v2/index.html)、[报告](run-20261112-v2/REPORT.md)、[实测速度图](run-20261112-v2/motion-continuity.png)。环绕51.22m、闭合误差7.5cm，再到第二通道04/5R旁停车；三次曲线接管时均约0.30m/s。七次Route递归与终末STOP检查通过。OCR近距离裁切冲突仍保留，不计为身份核验通过。

```bash
bash demo/warehouse_navigation/run_demo.sh \
  --output "/data/ticvla/outputs/navigation/codex-continuous-$(date +%Y%m%d-%H%M%S)" \
  --port 8802 \
  --decision-script demo/warehouse_navigation/decision-plan-v2.json
```

两次任务的首轮Route都包含整段任务一直到STOP，下一次输入保存上一次已接受的Route。第一段初始Route覆盖起始通道、远端右转、跨通道、再次右转、第二条通道返回、近端右转及回起点停车。第二段必须从本轮地图读出更远第二条通道的`04/5R`，首轮Route包含转向、横穿近端横通道、左转进入第二通道、接近目标及STOP。

当前实跑审核服务为`http://localhost:8802/review/`。复现时若8802仍被该服务占用，可将命令中的端口改为8803；输出目录也必须是新目录。

环绕由三段较长的五点曲线覆盖，剩余约1.8m时由独立决策进程提交下一段；控制进程在等待期间继续跟踪旧曲线，收到合法新曲线后按最新odom裁剪切换。只有任务完成或安全检查要求时停车，不再每2m等待新提案。第二任务的路线由本轮实走通道校验，目标坐标不从旧实验加载。原地大转向、未知空间及超时停车仍保留。

视频使用5fps采样最新RGB；地图与曲线按各自发布时刻显示。画面主要展示Semantic map、Landmark、Local、剩余Route、Waypoint及完整Instruction，背景和模型设置写入报告。这里的决策仍是Codex编写的代答逻辑；分进程调度支持以后接入真实LLM，但本演示不包含真实LLM推理延迟。

运行后可执行`python demo/warehouse_navigation/summarize_continuity.py --root <输出目录>`生成Route递归检查与切换时实测速度，不能只用视频流畅程度判断异步效果。旧v1结果保持不变。

```bash
bash demo/warehouse_navigation/run_demo.sh \
  --output "/data/ticvla/outputs/navigation/codex-demo-$(date +%Y%m%d-%H%M%S)"
```

默认模式等待 Codex 会话的源帧提案，不会自行伪造 VLA 输出。浏览器 `http://localhost:8800/` 显示 RGB、累计地图、局部规划曲线和状态。加 `--gui` 可同时打开 Isaac 窗口。

若提供本次执行后保存的 `decision-plan.json`，可复现已审核的 Codex 决策：

```bash
bash demo/warehouse_navigation/run_demo.sh \
  --output "/data/ticvla/outputs/navigation/codex-replay-$(date +%Y%m%d-%H%M%S)" \
  --port 8801 \
  --decision-script demo/warehouse_navigation/decision-plan.json
```

复现明确标记 `codex-reviewed-decision-replay`，并非新一次在线语言模型推理。第一阶段回放已审阅的环绕目标；第二阶段仍须从新运行的实际地图查出指定文字，利用本轮已走过的轨迹给出返回路线。目标缺失、文字歧义、目标已在视野内或控制超时会失败，不加载旧地图目标坐标。

启动时先原地观察一圈。几何/语义写入与控制分进程，控制读取最新RGB-D，不等待OCR；采集目标5Hz、语义采样1Hz。新脚本自动录制，并在结束后生成输出目录下的`review/index.html`。当前实跑由Codex逐次提交完成；自动决策回放入口未另跑第二个完整episode，不能保证每次OCR都得到同一目标。

已有结果可在仓库根目录执行`python -m http.server 8801 --directory demo/warehouse_navigation`，然后打开`http://localhost:8801/run-20261112-d/`。不要把本地HTML文件页当成已经发布的GitHub Pages。

## 数据与验证

- `navigation/requests/`：源观测、地图记录、任务及 Codex 响应、标签式 raw.txt。
- `navigation/decisions.jsonl`：实际接受或拒绝的提案，来源帧与原因。
- `navigation/curve-*.npz`：拟合曲线、源车体系与 odom 数组。
- `navigation/map/`、`landmarks.json`、`perception/`：地图数据库、地标、mask、OCR 与绑定结果。
- `loop-complete.json`、`target-task.json`、`task-finished.json`：分阶段证据，未发生不会补造。
- `capture/raw/`、`capture/trajectory.jsonl`、`navigation/executed-commands.jsonl`：真实采集与执行反馈。
- `report.json`：区分环绕、建图、目标到达与模型来源。

几何坐标与五点曲线测试见 `tests/warehouse/test_codex_demo.py`。任务是否实际成功以真实运行工件为准，不能把接口测试或决策回放称为训练 VLA 的自主导航能力。
