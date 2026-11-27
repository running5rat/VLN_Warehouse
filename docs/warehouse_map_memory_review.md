# 绕环建图后的超视距 Local / Route 审核

本实验先在原始仓库中沿既有轨迹绕环，建图 VLM 仅筛选候选实例（Landmark 入口），不输出 Local、Route 或 REFLINE。YOLO-E 分割、OCR 绑定和源帧三维投影继续运行。绕环结束后使用最后一个停车观测，提出一个此前已入图、当前 RGB 不可见的路标导航任务。此时仅调用 Qwen、GLM 或使用 Codex 会话生成 Local/Route，不启动导航控制器、AE 或训练。

这是理想仿真 odom 已知条件下的地图关系推理，不能称为 VLM 独立重定位。地图是带候选语义的累积障碍表面图，空白区域不保证可通行。蓝线是已发生的 odom 轨迹，不是任务的专家路线。保留身份与静态状态：跨帧文字确认不使地标成为静态锚点。

## 复现

从仓库根目录执行，输出目录必须是新目录：

```bash
bash DynaNav/warehouse_loop/run_realtime.sh \
  --selection-mode instances --landmark-variant original --headless \
  --port 8786 --output /data/ticvla/outputs/map-memory-new/loop

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.map_memory_review prepare \
  --loop /data/ticvla/outputs/map-memory-new/loop \
  --output /data/ticvla/outputs/map-memory-new/query
```

`prepare` 从真实地图中选择具有唯一读数且有不可见证据的 sign/label，优先跨帧身份已确认者；可以加 `--target '73/5R'`，但缺失、歧义、当前可见或深度证据不足会明确失败，不用模拟器路标目录补全。不可见检测使用已建图表面投影及当前同帧深度：完全在视锥外，或所有视锥内点被更近表面遮挡。它不是独立实例真值的可见性评测。

先核验 `query/input.json`、`current.png` 和 `observed-map.png`。API 请求使用这两张原始 PNG、相同提示和上下文；不会读取 `evaluation-only.json` 或 `target-evidence-*.png`。

以下命令向已有配置的 GLM/Qwen **各发一次收费请求**，沿用已有共享预算和固定推理配置，不自动重试：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.map_memory_review infer \
  --output /data/ticvla/outputs/map-memory-new/query \
  --configs /data/ticvla/outputs/stage1a/oracle-review-20261026/configs \
  --credentials /home/xinglinlu/.config/llm/credentials.env
```

Codex 组由会话查看同一份 `input.json` 及其中两张图片后产生提议，单独保存来源、耗时、原文和校验结果。不是 API 模型的替代评测，也不是人工批准标签。为避免互相影响，应先保存会话提议再阅读其他模型响应。

## 审核

结果页：`query/index.html`。文件包含各模型完整 `input.json`、`raw.txt`、API `response.json`、用量、时间、解析状态以及 `review-template.json`。网页可通过任意静态文件服务器查看，例如：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8787 \
  --bind 127.0.0.1 --directory /data/ticvla/outputs/map-memory-new/query
```

人工分别判断 Local 依据、目标标号保持、是否虚构当前可见、是否绕开货架、是否虚构任务完成。几何计算的目标相对位置只在评估区，不作为教师输入或完整路线 GT。语义正确率保持待审核；协议通过不代表导航通过。

2026-10-26 的首个运行保存在 `/data/ticvla/outputs/map-memory-review-20261026`。448 帧、约51.6 m，几何完整消费；本次并非实时性能验收，原运行的频率验收未全部通过。目标73/5R由三帧文字支持，当前226个已建图点均被更近深度遮挡。车辆绕环后结束仿真，后续从固定停车观测离线问答，没有发出导航命令。
