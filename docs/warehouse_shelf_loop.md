# DynaNav 货架编号与主车绕环建图

本轮已在宿主机 `/data/isaacsim` 的 Isaac Sim 5.0 实际运行。仓库沿用 `DynaNav/configs/data_collection/warehouse.yaml` 中的 NVIDIA `Simple_Warehouse/full_warehouse.usd`；未替换成简化方块场景。新增入口独立于默认 TIC-VLA / DynaNav pipeline。

## 已生成的场景

输出根目录：`/data/ticvla/outputs/warehouse-loop`。

- `scene/warehouse-labeled.usda`：完整仓库、28 个编号牌、Nova Carter 主车、车载 RGB-D 相机与俯视相机；可在 Isaac GUI 中使用 **File → Open** 打开。
- `scene/warehouse-labels-only.usda`：仅仓库和编号，用于 DynaNav 自己生成车辆的场景入口。
- `scene/dynanav-warehouse.yaml`：基于原仓库采集配置生成的场景配置。
- `scene/shelf-catalog.json`：28 个货架单元的编号、原始 USD prim 和评估位置；属于场景配置/评估数据，不输入识别或建图。
- `scene/textures/`：可重建的红白编号纹理。

原资产的 90 层托盘按相同 XY 范围归为 **7 排 × 4 个货架单元**。编号为 `01/1R` 至 `04/7R`，如 `03/2R` 表示第 2 排第 3 单元。每个单元一个身份，牌面中心高 2.55 m，尺寸 0.8 × 0.4 m，装在靠通道的上部立柱旁。牌的两面显示同一编号，面间距 12 mm，避免从不同方向回访时引入第二个货架身份。

GUI 直接打开 USD 后，主车不会自动绕环。需要执行下面的 `run.py`；添加 `--gui` 可显示实际行驶过程。

边行驶边看地图请使用新增的[单脚本实时建图入口](warehouse_live_mapping.md)，其中包含独立几何/语义进程、Isaac 地图面板和实际频率报告。本文后续 `validate.py` 命令仍用于离线验收。

## 实测结果

有效完整采集：`circuit-final/`；最终建图与验收：`acceptance/`。早期试跑和失败评估目录保留用于排查，不属于通过证据。

| 检查 | 实际结果 |
|---|---|
| 仿真引擎 | Isaac Sim `5.0.0-rc.45+release.23960.184afb15.gl` |
| 主车 | Isaac 5.0 Nova Carter；PhysX 轮关节速度控制 |
| 闭环路线 | `(-13,6) → (-13,27) → (-8,27) → (-8,6) → (-13,6)`，最后恢复朝向 +Y |
| 行驶距离 / 返回位置差 | 51.588 m / 0.143 m |
| RGB-D | 91 帧，960×720；来源时间跨度 90.217 s |
| 来源位姿与独立物理轨迹的最大差 | 0.0109 m |
| 已确认语义地标 | 8 个：`01–04/4R`、`01–04/5R` |
| 有效分割与建图证据 | 36 次；每个地标 2–8 个独立来源图像 |
| 已接受编号的身份正确率 | 36/36（固定场景工程验证） |
| 原始 YOLO-E mask 最小 IoU | 0.8464 |
| RGB 轮廓及深度平面精化后的 mask IoU | 0.9289–0.9907 |
| 单次观测牌面距离的最大 P95 | 0.00199 m，低于一个 0.05 m 配置栅格 |
| 融合地标中心最大位置误差 | 0.0373 m，阈值 0.15 m |
| 重复身份、重载、重复事件、无效来源、闭环回访 | 全部通过 |
| 回归单测 | 33 项通过 |

**覆盖范围**：安装编号的货架为 28 个，本次路线确认了其中 8 个。Oracle 中曾有至少 1,600 可见像素的编号共 12 个，另外 4 个没有通过此次识别链路。报告逐项列出未映射编号，不能将本次结果解释为整个仓库的语义覆盖完成。

**测量范围**：地标坐标代表编号牌观测表面的中心，不是货架体积的中心。RGB-D 来源位姿使用同步仿真位姿作为里程计；本轮未验证轮式里程计漂移或真实 SLAM。车辆使用固定航点控制器采集数据，不评价 LLM 导航、避障策略或训练效果。未启用行人随机化。

## 建图与验证链路

1. 冻结物理状态后，同一渲染批次取得 RGB、光轴 Z 深度、标定和相机位姿，使用冻结时刻的物理时间。避免稀疏渲染的 `reference_time` 被误当成物理采集时间。
2. RGB 中提取红白牌候选，透视矫正后由 **Tesseract 4.1.1** 逐字符识别编号。只使用牌面格式，不读取编号目录。数字位置允许 `O→0`、`A→4`、后缀数字位置 `S→5` 的 OCR 字形归一化，原始读数保存在事件 provenance 中。
3. OCR ROI 提示真实 **YOLO-E v8l-seg / Ultralytics 8.3.203**。拒绝与 RGB 牌面轮廓明显不符的 mask；随后与牌面轮廓求交，并用 RGB-D 的主平面剔除背景、支架深度点。保存原始和精化 mask，分别评估，不能用真值 mask 代替模型结果。
4. 同帧深度和来源位姿投影到现有 SQLite + 不可变 NPZ 地图。普通深度几何不受编号识别结果影响。
5. `fixed_shelf_markers=True` 是本固定安装场景的显式静态类别策略。首帧识别只建立 tentative 候选；相同文字与重叠三维几何获得第二个独立图像证据后，才确认静态地标。默认配置仍不信任 perception 提供的静态声明。同一几何出现冲突身份时明确拒绝，不生成第二个永久身份。
6. 所有写入决策结束后，评估器才读取独立 `supervision/` 实例标签和场景目录，检查身份、mask IoU、解析牌面几何误差、重复关联和回访。图像、目录、实例标签没有进入 LLM，也没有用于替代真实分割。

这是针对固定红白编号设计的工程实现，未宣称通用 OCR 或未知仓库的泛化能力。此轮没有训练模型。

## 复现命令

当前 Docker 容器内可使用宿主机启动包装器，无需切换终端或激活 Conda：

```bash
bash /project/docker/VLN/TIC-VLA/DynaNav/warehouse_loop/run_host.sh --check
bash /project/docker/VLN/TIC-VLA/DynaNav/warehouse_loop/run_host.sh \
  --gui --capture-hz 1 \
  --output /data/ticvla/outputs/warehouse-loop/manual-01
```

包装器通过当前容器已有的宿主机命名空间访问，以宿主机用户 `xinglinlu` 运行 Isaac 自带 Python，自动设置工程目录、清理容器的库路径并连接宿主机桌面。已检查宿主机运行时为 glibc 2.35；容器自身为 glibc 2.31，直接在容器调用 `/data/isaacsim/python.sh` 会出现 `GLIBC_2.34` / `GLIBCXX_3.4.30` 缺失。无需更换容器的系统 libc。

输出目录必须不存在。该入口新建 Isaac 窗口并自动行驶采集，不接管已打开的 GUI。默认宿主机路径、用户、桌面可分别通过 `TICVLA_HOST_REPO`、`TICVLA_HOST_USER`、`TICVLA_DISPLAY`、`TICVLA_XAUTHORITY` 覆盖。

2026-09-29 修复：桌面显示号会随重启改变（本机曾从 `:0` 变成 `:1`）。启动器现在读取当前用户的桌面会话，并使用 `XOpenDisplay` 实际检查连接；`--check` 也执行此检查。显式指定错误显示号会在启动 Isaac 前报错，不会回退到其他桌面。旧版仅检查 Xauthority 文件可读，不能保证窗口能够创建。

修复后已实际执行 `--gui --max-steps 180`：X11 窗口状态为 `IsViewable`，车辆行驶超过 1 m，写出 4 帧同步 RGB-D，并以退出码 0 结束。记录位于 `gui-fix-check-20260929/gui-launch-check.json`，完整日志为 `gui-fix-check-20260929.log`。新增 4 项桌面选择回归测试通过。此短程检查不替代之前的完整绕环验收。

本次首次 GUI 初始化等待 RTX 着色器编译约两分钟，Kit 日志中可见 `Waiting for RtPso async group async compilation`。日志中的 Warp `cuDeviceGetUuid` 消息在成功试跑时仍出现；NVIDIA 记录了 [Warp / R580 的对应兼容问题](https://github.com/NVIDIA/warp/issues/851)，本次没有为此更换 Isaac、驱动或系统库。

宿主机执行采集，使用未存在的输出目录。不要用容器内较旧的 glibc 启动 Isaac。

```bash
cd /home/xinglinlu/autodriving_ws/docker/VLN/TIC-VLA
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME \
  PYTHONPATH="$PWD" /data/isaacsim/python.sh \
  DynaNav/warehouse_loop/run.py \
  --output /data/ticvla/outputs/warehouse-loop/reproduce-01
```

加 `--gui` 显示窗口；加 `--build-only` 只导出编号场景与主车。已有 GUI 会话和原仓库资产不受脚本修改。

容器中执行离线识别、建图与评估：

```bash
cd /project/docker/VLN/TIC-VLA
PYTHONPATH=. /data/ticvla/envs/warehouse-io/bin/python \
  DynaNav/warehouse_loop/validate.py \
  --capture /data/ticvla/outputs/warehouse-loop/reproduce-01 \
  --output /data/ticvla/outputs/warehouse-loop/reproduce-01-map \
  --weights /data/ticvla/models/yoloe-v8l-seg.pt \
  --host-tesseract
```

`--host-tesseract` 通过 `nsenter` 使用宿主机已有 Tesseract，PNG 经 stdin 传输，无需在 Isaac Python 中安装分割/OCR 依赖。在其他环境中安装 Tesseract、FFmpeg 后可省略该参数。Python 依赖沿用 `requirements-warehouse-io.txt` 和工程原有 Matplotlib。

加载本次已保存的地图并导出当前 MapView：

```bash
PYTHONPATH=. /data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.mapping.cli \
  --store /data/ticvla/outputs/warehouse-loop/acceptance/map \
  export --image-id circuit-final:00000090 \
  --output /data/ticvla/outputs/warehouse-loop/reloaded-view
```

地图目录包含 SQLite 和不可变几何 NPZ；保留 `circuit-final/raw/`，以便来源图像哈希、深度和事件可追溯。

```bash
PYTHONPATH=. /data/ticvla/envs/warehouse-io/bin/python \
  -m unittest discover -s tests/warehouse -p 'test_*.py'
```

单测中的微型优化器回归检查沿用原测试套件，不是本轮模型训练。

## 查看产物

- `acceptance/semantic-map.png`：全程实际观测几何、车体轨迹、8 个编号地标，绘图不使用 Oracle 填充未观测区域。
- `acceptance/global-map.npz`：累计三维点云和来源轨迹；不将此历史点云视为实时安全占用地图。
- `acceptance/map/map.sqlite`、`acceptance/map/arrays/`：持久地图及几何。
- `acceptance/final-mapview/`：末帧局部 MapView，继续可用于已有 LLM 图像地图接口。
- `acceptance/mapping.mp4`、`acceptance/overview.mp4`：车载分割叠图和仓库俯视视频；H.264，采样画面按 6 fps 播放，约为 6 倍速。
- `acceptance/overlays/`、`acceptance/masks/`：逐帧叠图、原始 YOLO-E mask 与精化 mask。
- `acceptance/report.json`、`acceptance/events.json`：逐项门槛、真实测量和事件处理结果。
- `circuit-final/trajectory.jsonl`：独立控制轨迹、时刻、航点和轮速控制对应的 v/ω 命令。
- `unit-tests.log`：回归测试结果。

实现入口：`DynaNav/warehouse_loop/{scene,run,validate}.py`，识别与深度精化：`ticvla/warehouse/mapping/markers.py`。Nova Carter 采用 Isaac 5.0 官方差速轮控制接口：[Mobile Robot Controllers](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/robot_simulation/mobile_robot_controllers.html)。
