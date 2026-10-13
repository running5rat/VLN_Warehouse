# v0.6 导航联调（无训练）

本入口把完整 VLM、参考线、语义快照、九模态 Flow、候选检查和 Isaac 轮速控制接在一起。参考线头、KV 压缩头和 Flow 均为固定种子初始化，**没有经过导航训练**。`model` 不保证运动；候选全部被拒绝时停车。`geometry-baseline` 只使用已观测空间规划，其移动不代表模型已学会导航。

## 启动

在项目根目录运行；脚本固定使用 `/data/ticvla/envs/warehouse-realtime/bin/python` 执行模型和地图进程，再通过已有 `run_host.sh` 在宿主 `/data/isaacsim` 启动 Isaac Sim 5.0。无需先打开 Isaac，也不要在旧容器 Python 中直接加载 Isaac。

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --task-mode explore --execution-backend model \
  --output "/data/ticvla/outputs/navigation/model-explore-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

浏览器打开日志中的 `http://127.0.0.1:8766`。远程运行可用 SSH 本地端口转发。`--headless` 关闭 Isaac 窗口，浏览器仍可使用。`--max-steps 3600` 表示 60 秒物理步进预算，不包括模型预热、场景加载和停车后的结果归档。输出目录必须是新目录。

工程运动对照显式改为 `--execution-backend geometry-baseline`。程序不会自动从模型模式切换到几何对照。原 `run_live.sh` 建图与固定绕环入口不变。

### 地标任务

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --task-mode landmark --execution-backend geometry-baseline \
  --target-landmark '71/4R' \
  --map /data/ticvla/outputs/warehouse-loop/instances-original-20261012-154216/mapping/map \
  --map-anchor-image instances-original-20261012-154216:00000000 \
  --output "/data/ticvla/outputs/navigation/landmark-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

`--map-anchor-image` 是操作者指定的旧地图重叠观测，在新 episode 首帧用深度几何验证 SE(2)；不默认坐标一致，也不是全局重定位能力。以上命令针对相同初始场景。验证失败时旧子图保持独立。配准结果保存在 `navigation/map-registration.json`。复制旧地图到新目录后才更新，不修改原地图。重复文字必须改用明确的 `landmark_uid`。

目标为物体旁的已观测自由邻域。局部可通行区域过短时只走已验证的短前缀，不把前缀终点报告为到达目标。

### 行人跟随

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --task-mode follow --execution-backend geometry-baseline \
  --output "/data/ticvla/outputs/navigation/follow-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

在浏览器冻结当前 RGB，并拖框选中行人。服务绑定显示图像的 `image_id` 和像素框，随后用真实 YOLO-E `person` 检测和同帧深度追踪。也可以传入 `--target-roi roi.json`：

```json
{"image_id":"本次输出目录名:00000000","bbox":[430,315,530,720]}
```

框必须人工核验并属于指定图像，不能把另一帧检测结果直接绑定到当前帧。目标长时间丢失后，需要在重新出现的图像上明确重确认；不会自动换人。

本轮场景使用官方人形 USD，以 0.2 m/s 平移，尚未加入步态动画。首次有效 TEXT 后开始移动，在 15–18 秒暂时隐藏以测试丢失停车。场景控制器的身份和位姿不传入感知或规划。该测试只支持单人短时跟踪，不证明长期身份保持。

## 输入、输出与执行路径

1. `run.py` 同步采集 RGB-D、标定、仿真 odom 和源单调时钟；导航分支只读取受 watchdog 检查的命令，使用差速轮控制。
2. `navigation/runtime.py` 是唯一地图写入者，消费所有源帧。导航局部栅格为 5 cm，深度采样 stride 32，保留 2 秒局部安全历史；持久观测仍全部保存。几何不等待模型。
3. `navigation/vlm.py` 输入当前及 1/2/3 秒历史图像、渲染 MapView、有界地标记录、任务和执行状态。每请求只执行一次 TEXT 或一次 PREFILL，没有先做自由物体问答。
4. TEXT 按固定标签顺序输出 `<landmark>…</landmark><local>…</local><Route>…</Route><waypoint>…</waypoint><REFLINE>`；本入口使用兼容的 `image_id/category` 两字段地标记录。真实模型在语法约束下生成内容，不补造解析成功样例。最多 2,048 tokens。
5. `<REFLINE>` 最后一个 token 真正进入语言模型后，读取 hidden state，经 Decoder 得到参考线；同次 KV 经压缩器得到 `[16,512]`。PREFILL 不生成文本、不更新 Route，也不发静态地标事件。
6. `navigation/core.py` 原子提交 Route 和快照，验证任务、基础版本、来源时钟、epoch、乱序、张量形状和跟随绑定。旧参考线按 odom 对齐并裁剪；`z_sem` 不旋转。参考线原先可通行而当前地图出现阻挡时，快照失效。
7. `navigation/flow.py` 直接消费 `z_sem`，共享条件编码和 3 个 Transformer block，512 hidden、8 heads。横向 3 × 纵向 3 个模态，H=30、dt=0.1 s、4 步 Euler，输出 `[B,9,1,30,2]`。旧单候选训练类保持不变。
8. 规则 Scorer 先拒绝非有限、超速、过大加速度、转向不可执行、碰撞和未知区域候选，再按进展、偏离、平滑性排序。Pure Pursuit 使用短前缀和速度前馈。Isaac 端另做命令期限、传感器年龄、epoch、取消和实际运动弧线检查。
9. 非跟随 TEXT 的 Landmark 指定源帧驱动 YOLO-E 和学习型 OCR；Local 只保存为待验证关系，不修改 TF。新地标仍为 tentative。跟随使用独立 person 词表，不修改旧建图引擎。

## 时间和结果解释

物理 60 Hz，RGB-D/几何目标 5 Hz，Isaac 命令检查目标 30 Hz；Flow 目标 5 Hz，PREFILL 目标 1 Hz。TEXT 最大来源年龄 10 秒，PREFILL 3 秒；超时如实拒绝。这些是低速联调配置，不是已达到的实时性能承诺。

`navigation/metrics.jsonl` 记录服务耗时、源年龄、队列替换、候选拒绝及命令；`executed-commands.jsonl` 是 Isaac 实际执行命令。不要把请求频率、模型 GPU forward 时间或重复显示次数当作完整处理频率。

每个实验保存 `config.json`、环境版本、GPU 采样、独立地图、VLM 原文和 prompt、快照 NPZ、Flow 轨迹、候选评分、真实控制命令、同步采集和 `report.json`。`index.html` 要通过 HTTP 打开；实验结束后可运行：

```bash
python -m http.server 8766 --directory /data/ticvla/outputs/navigation/某次实验
```

浏览器的源图、参考线、九模态候选、实际执行来源和地图状态分别展示。紫色为实际 odom 轨迹，橙色为模型参考线，浅蓝为模型候选，绿色为当前选中执行路径；几何对照的绿色路径不会覆盖模型原始候选文件。

## 离线真实模型复现与测试

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.navigation.replay \
 --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
 --frame 200 --output /data/ticvla/outputs/navigation/offline-new

OMP_NUM_THREADS=2 /data/ticvla/envs/warehouse-realtime/bin/python \
 -m unittest discover -s tests/warehouse
```

离线跟随使用明确标记的协议夹具，不算真实行人跟踪验收。在线三种任务和两种控制来源必须分别报告。所有新头权重、配置和哈希随实验保存；接口成功不意味着模型质量验证通过。

## 后续差距

本轮不实现训练、学习型 MapEncoder、历史视觉 token 压缩、学习型 Scorer、多机器人、精确停靠或取货。无人为替换模型参考线、Oracle 路线或隐藏目标位姿。永久静态性、长期身份保持、任务成功率和部署实时性能均未验收。实际已通过与未通过项目以本轮验证报告为准。
