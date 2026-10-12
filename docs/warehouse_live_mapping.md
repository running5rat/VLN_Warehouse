# Isaac Sim 行驶时增量语义建图

## 单脚本启动

在当前 Docker 工程终端运行（无需激活 Conda；输出目录必须是新目录）：

```bash
bash /project/docker/VLN/TIC-VLA/DynaNav/warehouse_loop/run_live.sh \
  --output /data/ticvla/outputs/warehouse-loop/live-manual-01 \
  --keep-viewer
```

脚本通过 `run_host.sh` 在宿主机 `/data/isaacsim/python.sh` 启动 Isaac Sim 5.0 GUI；通过容器 `/data/ticvla/envs/warehouse-io/bin/python` 启动建图和独立 YOLO-E 推理进程。当前机器使用宿主用户 UID/GID 1000；其他机器可设置 `TICVLA_HOST_UID/GID`、`TICVLA_HOST_USER/REPO`。这个入口针对已配置好的当前 Docker/宿主共享 `/data` 部署。

Isaac 内自动出现 **Live Semantic Map** 面板，可拖动或停靠到视口旁。主视口仍可切换 `/World/MainRobot/chassis_link/MappingCamera` 或 `/World/Overview`。浏览器额外视图为 `http://127.0.0.1:8765`，作用域是运行脚本的网络命名空间；远程访问可使用端口转发，或显式设置 `--bind 0.0.0.0 --port 8765` 并按实际网络发布端口。`--keep-viewer` 在采集和建图结束后保留网页，Ctrl+C 退出。Isaac 在绕圈完成后自动关闭。

短运行检查：添加 `--max-steps 180`，只验证约 3 秒仿真运动，不代表完整地图覆盖。`--headless` 可关闭 GUI，此时不能验收内嵌面板显示。

## 调度与数据边界

- 物理/控制：每步 1/60 秒仿真时间。
- 视口：每两个物理步显式渲染一次，目标 30 次/秒。
- RGB-D：每 0.2 秒仿真时间采集一次，目标 5 Hz；调用 `World.render()` 刷新物理变换，再由 Replicator `step(delta_time=0, pause_timeline=False, wait_for_render=True)` 明确调度并等待本次传感器帧完成，获取同帧 RGB、光轴 Z 深度、内参和相机源位姿，避免每次 pause/play。断言采集前后仿真时间不变，并对照 PhysX 检查源位姿。采集同时完成所在时隙的视口刷新，避免重复渲染。写完文件后提交 JSONL，追加独立墙钟时间用于延迟测量。
- 几何：持续读取完整 JSONL 记录，处理所有 RGB-D，投影并保存到 SQLite + 不可变 NPZ，不依赖识别结果。
- 语义：每 1 秒仿真时间选择一帧，独立进程执行 RGB 牌面候选、OCR、真实 YOLO-E 和 mask/深度平面检查。结果由唯一地图写入进程完成重复观测静态确认及关联。推理耗时超过采样间隔时排队，不丢弃已经选定的语义帧。
- 显示：几何或语义结果变化时发布地图图片/状态，Isaac 主线程更新 UI。网页显示最新语义来源画面，可能落后于几何，状态显示其源时间与队列长度。

实时模式停止额外生成 Oracle 实例标签和俯视相机录像，采集渲染子帧从离线 3 调为 1；现有离线默认采集仍保留 3 子帧和 Oracle。在线地图是**已观测障碍表面的累积 BEV**，不把历史占用图冒充动态避障或自由空间规划图。蓝色为轨迹，红色为源帧机器人位置，橙色为暂定地标，绿色为已确认地标；地标位置仍是编号牌观测表面中心。该实时入口不调用 LLM、不训练。

在线 PNG 使用无压缩等级 0，地图 NPZ 使用无损压缩等级 1，减少 CPU 开销；会比离线默认压缩等级 6 占用更多磁盘，但图像像素、数组数值和加载格式不变。不要删掉上述同步等待：实测仅调用 `render/get_data`，甚至连续刷新多次，也可能取得旧的 annotator 缓冲或初始相机参数，导致移动视角的地图错位。

RGB-D 返回主线程后复制缓冲，交给容量为 4 帧的后台写盘线程；按原顺序写完 RGB/depth 后才提交 `frames.jsonl`。队列满时施加背压，写盘异常向主进程传播，结束时等待落盘完成。该线程不访问 Isaac/PhysX/UI；几何和语义仍在独立 Python 进程中运行。

## 性能报告

- `report.json`：联合结果、完整绕圈状态、各模块实测吞吐和是否达到目标。
- `capture/performance.jsonl`：物理步、显式渲染、同步采集、地图面板更新的逐次墙钟时刻与服务时间。
- `capture/performance-report.json`：仿真与现实耗时、实时因子、采样频率。
- `capture/writer-performance.jsonl`：后台写盘的实际完成次数和耗时。
- `mapping/metrics.jsonl`：几何、语义和地图发布计时、OCR 耗时、真实 YOLO 调用次数。
- `mapping/report.json`：包括排队清空阶段在内的建图统计、事件处理结果和来源时间。
- `map-validation.json`：建图结束后独立读取评估目录，检查地标身份与位置、投影误差、重复证据、源位姿与物理轨迹一致性。当前在线运行不采集 Oracle mask，因此不报告本轮 mask IoU。
- `mapping/map/map.sqlite`、`mapping/map/arrays/`：可恢复地图。
- `mapping/map.png`、`mapping/semantic.png`、`mapping/masks/`：地图和真实分割结果。
- `isaac.log`、`mapping.log`：独立进程日志。

“达标”要求实际墙钟吞吐至少达到目标的 95%，且样本不少于 10 次。几何和语义的联合报告仅统计车辆运动窗口内的完成次数，不能靠结束后的集中补算冒充在线达标。报告中的 `capture_source_hz` 是仿真时间采样率，不能作为现实吞吐的替代。`viewport_render` 记录调度的视口更新完成频率，采集共用其中的时隙，不是 GPU 实际屏幕 present 计数。每项同时记录 P95 间隔，平均频率达标不表示每个间隔都均匀。

语义 Hz 表示处理的来源帧频率，包含没有可识别编号的帧。没有候选时不会伪造 YOLO 调用；`yolo_calls` 单独统计真正分割次数。实测不足时报告保持失败，不自动降低目标。

## 本机实测：2026-09-29

有效运行目录：`/data/ticvla/outputs/warehouse-loop/live-acceptance-20260929`。使用宿主 Isaac Sim 5.0、RTX 4090，同时开启 GUI、地图面板、后台 RGB-D 写盘、增量几何和真实 OCR/YOLO-E。

| 模块 | 目标（Hz） | 实际平均（Hz） | 95% 目标门槛 |
|---|---:|---:|---|
| 物理/控制步进 | 60 | 57.44 | 通过 |
| 视口更新 | 30 | 28.72 | 通过 |
| RGB-D 采集 | 5 | 4.79 | 通过 |
| RGB-D 写盘 | 5 | 4.79 | 通过 |
| 几何建图（运动窗口内） | 5 | 4.78 | 通过 |
| 语义来源帧处理（运动窗口内） | 1 | 0.953 | 通过 |
| 地图面板新图更新 | 5 | 4.19 | 未达到 |

物理步长仍为 1/60 秒仿真时间；表中为现实时间内完成的次数。核心 30/5/1 Hz 按 5% 容差通过，但不能说所有模块严格达到了标称频率，`all_measured_module_rates_met=false`。视口更新间隔 P95 为 0.103 秒，受同步采集阻塞影响，不能将平均 28.72 Hz 理解为均匀的 30 FPS 呈现。

- 完整绕圈 51.588 m，返回位置差 0.143 m；仿真推进 90.217 秒，运动窗口现实耗时 93.350 秒。
- 448 帧 RGB-D 全部进入几何地图；91 帧进入语义处理，真实 YOLO-E 调用 34 次。未丢几何帧或选定的语义帧。
- 几何从文件提交到处理完成的 P95 延迟 0.271 秒；语义最大队列长度 2（包含正在处理的帧）。
- 确认 6 个编号地标：`01/4R`、`02/4R`、`03/4R`、`01/5R`、`02/5R`、`03/5R`。`04/4R`、`04/5R` 未确认；不满足此前离线验收的八编号完整覆盖。
- 已确认地标的身份、重复证据、坐标和表面投影检查通过。最大编号牌中心位置误差 0.0353 m，最大表面 P95 距离 0.000366 m；源位姿与独立物理轨迹最大差约 0.0109 m。当前在线运行未采集 Oracle mask，未重新计算 mask IoU。
- 回归测试 43 项通过；新增测试覆盖不完整 JSONL 提交、墙钟频率判定、补算不能冒充实时吞吐、后台缓冲复制/写入错误传播和快速 NPZ 无损重载。

`report.json` 中 `map_validation_pass` 仅表示本轮已观测地标的几何正确性检查；`route_coverage.complete=false` 单独表示编号覆盖未完成。旧的 `live-smoke*`、`live-render*`、`live-final*`、`live-flush*` 等目录是开发实验，不能当作最终通过证据。同步优化前的完整对照运行 `live-verified-20260929` 地图正确、八编号覆盖完整，但视口约 27.46 Hz、几何 4.58 Hz、语义 0.91 Hz；速度和识别覆盖须分别报告。

实际 Isaac 窗口截图：`isaac-window.png`；当前地图：`mapping/map.png`。地图窗口可以拖到右侧，或手动停靠，避免挡住车辆视口。
