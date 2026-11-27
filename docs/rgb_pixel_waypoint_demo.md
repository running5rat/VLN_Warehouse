# 源 RGB 像素 waypoint 与两任务 Oracle 对照

本补充落实 2026-11-12 用户要求，作用于仓库 demo；不修改已有训练数据中的米制 waypoint。

`rgb-pixel-waypoint-v1` 的五个 `[u,v]` 是 **960×720 原始 RGB 像素**，左上为原点，u 向右、v 向下。模型只提出图内可见地面上的局部路线前缀；全局目的地和未完成步骤保留在 Route 中。未知方向或停车允许 `[]`。画面内不等于可通行，标牌上的像素不能作为车辆驶入位置。

输入是当前 RGB、同帧分割叠图、此前已经提交的语义地图和递归 Route；报告同时记录实际输入图像、地图截止时间、源位姿与文件哈希。离线与在线入口提供上一有效 Local 和相对运动；超过30秒的 Local 失效为 unknown。图层样式分别记录，不宣称两次运行的地图状态相同。

输出 JSON 仅含 `landmark / local / route / waypoint`。Landmark 至多三个当前检测类别，Local 90 字符以内，Route 600 字符以内，保持未完成步骤与最终 STOP。任务停止只能根据实际停止及完成证据确认。完整冻结 prompt 保存在每次实验 `system.txt`；模型原文不被程序修复。

执行路径：像素 → 同帧深度光轴 Z → K 逆变换 → 相机到车体 → 源 odom → 工程 PCHIP 曲线 → 当前车体系剩余段 → Pure Pursuit → 现有 footprint/占据/命令时效检查。平地仓库中地面高度容差 15 cm；原始深度不可用时明确拒绝，不补地平面假深度。像素越界、拟合曲线出视野、连接曲线被遮挡均拒绝。车头至第一个可见点之间可能有相机近端盲区，仍依靠已观测地图和 footprint 检查，不宣称这个连接段来自当前 RGB。

工程曲线不是 `<REFLINE>` hidden state Decoder 或 Flow 输出，本轮不运行这两者。旧 `legacy-body-metric` 数据和实跑视频保留原含义。

在线服务每 2 秒调度一次，保留一个处理中请求与一个最新等待帧；不以旧曲线到达末端作为请求条件。跟踪进程独立工作，收到新有效曲线立即按当前 odom 切换。源年龄超过 5 秒的回答归档，不发布控制；旧像素曲线超过 5 秒也失效停车。无效或空的新曲线不能继续冒用旧指导。有效文本的递归与几何接受分开。RGB Oracle 模式中，地标由回答类别驱动同源缓存 mask 入图，不使用旧的固定登记类别策略补救模型失败。

在线接口（实验性，尚未完成新协议 Isaac 闭环验收）：

```bash
bash demo/warehouse_navigation/run_demo.sh \
  --rgb-oracle codex-session \
  --instruction "Follow the starting aisle to its far end, turn right across the cross-aisle, turn right into the second aisle, return to the starting point and STOP." \
  --output /data/ticvla/outputs/navigation/rgb-session-new --port 8804
```

`navigation/rgb-oracle/rgb-XXXXXX/input.json` 是冻结请求，会话只在同目录写 `oracle-answer.json`。不能写入位姿或地图坐标替代像素。`--rgb-oracle qwen3.7-plus` 使用现有北京端点和预算账本，是真实付费请求；不应因离线格式可用就宣称已可实时导航。该入口目前不自动执行两任务切换或把一段 STOP 文本视为完成，需要独立任务完成核验。

对照报告命令：

```bash
PYTHONPATH=. /data/ticvla/envs/warehouse-io/bin/python \
  demo/warehouse_navigation/rgb_oracle_review.py prepare \
  --source /data/ticvla/outputs/navigation/codex-continuous-20261112-v2 \
  --output /data/ticvla/outputs/navigation/rgb-oracle-new
# qwen 子命令调用 API；读取 credentials.env，使用持久预算保护，无静默重试。
```

本次报告是两任务固定轨迹回放，37 帧（含两次真实任务结束帧），并非这次像素预测重新驱动车辆。共同地图来自原真实感知的因果重建，不是两模型各自形成的新闭环地图。Codex 是会话视觉代标注，含既有讨论上下文，不能当作无历史信息的 API 盲测。视频按源时刻展示回答，不代表其实际网络延迟；报告单列请求往返耗时、失败和未执行项。

审核视频服务应使用支持 HTTP Range 的入口，避免浏览器无法跳转进度；普通 `python -m http.server` 在本环境会忽略 Range 请求。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.video_server \
  --directory /data/ticvla/outputs/navigation/rgb-oracle-20261112-v1 --port 8803
```

`Projection: projected` 仅表示深度投影与曲线可见性检查通过，不表示 Route 一致或控制可执行。第437帧的 Codex 点全部在图像中线，投影为前方3.35至13.65米的直线；录制运动在前方约4米处已经右转。该例应保留为路线问题案例，不能作为成功转弯标注。单帧控制诊断也因 `shield_occupied_or_unknown` 返回停车；此检查不单独证明具体碰撞位置。

待闭环验证的关键问题：纯前视像素协议没有表达原地转向的动作通道。需要转向看见新区域但 `waypoint=[]` 时，当前控制器停车，不能用隐藏坐标、画面外像素或关键词脚本悄悄完成转弯。该接口缺口与 Qwen 的推理时延必须单独处理后才能宣称完成像素路线自主环绕。

## 仅目标指令的 Codex 会话重标注

`/data/ticvla/outputs/navigation/codex-goal-only-20261112-v2` 保留两任务37帧，第二任务初始化仅为 `Go to sign 04/5R and STOP beside it.`。不提供第二通道或转向提示；后续递归使用新输出的 Route/Local。没有调用 Qwen 或 GPT API。Codex会话已知道此前演示，因此属于非盲的会话辅助标注，不能冒充独立API模型能力测试。

历史地图按相同因果观测重绘为编号点与图内图例，保持OCR冲突和对象tentative状态；未加入未来计划。视频分别显示 Previous Route 与 Generated Route。24帧非空投影通过、13帧空点；所有24帧单帧控制诊断均因占据或未知区域拒绝运动，不能将投影成功当作控制通过。两个任务仍是旧运动记录回放。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.video_server \
  --directory /data/ticvla/outputs/navigation/codex-goal-only-20261112-v2 --port 8804
```

输入准备：`demo/warehouse_navigation/codex_goal_only_review.py`；本次会话标注的固定序列化记录：`codex_goal_only_labels.py`；验收报告：`finalize_goal_only_review.py`。固定标签脚本仅用于复现本次标注文件，不是可以为新场景推理的VLM。
