# 原始仓库语义建图 Demo

已在 Isaac Sim 5.0 上重新执行 `original` 场景完整一圈。用户所称的 `origin` 在现有脚本中对应 `--landmark-variant original`。

结果入口：[原始场景视频与地图](original-20261013/index.html)。详细结果：[本次实验说明](original-20261013/REPORT.md)。

## 查看

可直接用浏览器打开 `original-20261013/index.html`。也可以从仓库根目录运行：

```bash
/data/ticvla/envs/warehouse-realtime/bin/python -m http.server 8791 \
  --bind 0.0.0.0 --directory demo/semantic_mapping
```

本机访问 `http://localhost:8791/`；其他电脑访问 `http://<运行主机IP>:8791/`，容器网络需能访问该端口。

## 一条命令重新执行和打包

在当前仓库根目录、可进入宿主命名空间的容器 root 环境中执行：

```bash
bash demo/semantic_mapping/run_demo.sh
```

脚本使用 `/data/ticvla/envs/warehouse-realtime/bin/python` 跑感知和录制，通过已有 `run_host.sh` 使用宿主 `/data/isaacsim/python.sh` 跑 Isaac Sim 5.0。需要现有仓库资产、模型、TensorRT 引擎和 ffmpeg。每次创建独立输出目录，拒绝覆盖既有运行。

- 完整原始数据：`/data/ticvla/outputs/warehouse-loop/demo-original-<时间>/`。
- 视频、地图、报告：`demo/semantic_mapping/original-<时间>/`。
- 可选位置参数：`bash demo/semantic_mapping/run_demo.sh <新原始数据目录> <新交付目录>`。
- 录制采用 headless Isaac 的真实车载 RGB 和在线发布的地图/分割画面；不录桌面 UI。
- 不启动完整结构化导航推理、Flow 或模型控制，车辆仍沿原有固定绕环轨迹运动。

仅想打开 Isaac GUI 与实时建图网页，使用原入口：

```bash
bash DynaNav/warehouse_loop/run_realtime.sh \
  --selection-mode instances --landmark-variant original \
  --output "/data/ticvla/outputs/warehouse-loop/original-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

## 工件

| 文件 | 内容 |
|---|---|
| `mapping_live.mp4` | 95 秒实时组合录像：RGB、累计地图、全部检测、入图实例；包含模型滞后与等待 |
| `isaac_camera.mp4` | 448 帧原始 RGB、5 fps、89.6 秒；按仿真采样时间编码 |
| `semantic_bev.png` | 后处理地图、车辆轨迹、地标索引与采用文字 |
| `semantic_3d.png` / `semantic_points.npz` | 三维观测表面及轨迹；背景点云为显示采样，不替代完整地图 |
| `map.png` | 最后一次在线发布地图 |
| `map.tar.gz` | 完整 `map.sqlite` 和不可变几何数组；解压后可直接用 MapStore 加载 |
| `landmarks.csv` / `landmark-view.json` | 包含隐藏对象的全部地标、位置、文字和状态 |
| `postprocess/` | 后处理前后对照、子串融合与冲突记录 |
| `samples/` | 源 RGB、mask、三维数组、文字裁剪及可视化样例 |
| `evidence/` | 完整 VLM 请求回答、逐帧处理记录、模型来源、时延与运行日志 |
| `scene/` | 场景层、纹理、配置和目录；USD 基础资产仍使用本机现有路径 |
| `summary.json` / `verification.json` | 本次结果及视频、地图加载、文件完整性检查 |
| `video-timeline.jsonl` | 组合视频各帧的墙钟、RGB 源帧和检测/OCR 源帧 |
| `manifest.json` | 打包时文件尺寸及 SHA-256；后续生成的验证说明另行保存 |

`map.tar.gz` 约 833 MiB，保留完整历史几何；视频和大文件已存到工作区，没有执行 git add、commit 或 push。

## 地图加载

```bash
mkdir -p /data/ticvla/outputs/demo-map-restored
tar -xzf demo/semantic_mapping/original-20261013/map.tar.gz \
  -C /data/ticvla/outputs/demo-map-restored

/data/ticvla/envs/warehouse-realtime/bin/python - <<'PY'
from ticvla.warehouse.mapping import MapStore
store = MapStore('/data/ticvla/outputs/demo-map-restored/map')
print('version:', store.version)
print('observations:', len(store.records('observations')))
print('landmarks:', len(store.records('landmarks')))
store.close()
PY
```

归档中的几何和地标可独立加载；数据库保留原始 RGB-D 绝对路径用于来源核验。重新执行分割/OCR 或源图哈希核验需保留完整原始运行目录。本次目录为 `/data/ticvla/outputs/warehouse-loop/demo-original-20261013-1133`。未将全部 RGB-D 复制进演示目录。

## 版本与结果边界

基于 `7b97f85622de91674327166f6f84d5390059a306` 的建图实现，实际运行仓库 HEAD 为 `9cbc125e374c6a1272a4d2edf32bf1dda395be0c`。核对的建图入口、实例调度、地图和感知核心与基准 commit 无差异。当前 HEAD 含导航扩展，但此次未启用导航分支；版本与模型信息保存在结果中。

这是一轮真实建图演示与工件验证。全部对象的静态状态仍为 `tentative`；文字身份确认与静态确认分开。存在误分类、无关文字和延迟超标，不据此宣称识别准确率、长期唯一身份或实时性能达标。
