# 完整 R2R 训练集的数据准备

2026-11-12 用户确认：先完成数据准备，暂不调用 API。当前入口仅使用本地 Habitat 与几何处理，不读取凭据、不发送教师请求、不训练。

## 范围与资源核验

本地 `NaVILA/data/datasets/R2R_VLNCE_v1-3_preprocessed` 已包含完整数据，无需重复下载：

| 项目 | 数量 |
| --- | ---: |
| 原始 train 任务 | 10,819 |
| train 建筑 | 61 |
| 任务对应的完整回放帧 | 642,063 |
| 精确不同的回放轨迹 | 3,603 |
| 去除等价回放后的 RGB-D 帧 | 213,797 |

61 个场景的 `.glb`、`.navmesh` 均存在，所有任务有动作/轨迹 GT。源数据和场景文件哈希保存在 `source-inventory.json`。官方 val_seen、val_unseen、test，以及 EnvDrop 增广集不加入本轮训练输入；一个任务不是一个建筑。

图像共享要求场景、起点、朝向、动作序列、GT位置和相机配置完全相同，不能仅凭 reference_path 相似复用。以硬链接共享 RGB-D 文件；各任务分别保存 instruction、task/image ID、odom epoch、verification-only 和后续标注上下文。原有四个比较任务的同路线任务组继续标为比较用途，从汇总训练 manifest 排除，仍准备并统计它们。

相机沿用既有 R2R 配置：512×512、水平视场90度、高1.25m、光学 Z 深度0–10m、理想局部 odom。每个离散动作占0.2秒是合成回放时钟，不是实际采集速度或 IMU 测量。

## 执行

工作目录为仓库根目录。初始化命令只能用于一个新目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.dataset_tools.r2r_full_prepare catalog \
  --output /data/ticvla/datasets/r2r-full-20261112 \
  --reuse /data/ticvla/datasets/stage1a-20261021/episodes-v4 \
  --previous-selection /data/ticvla/datasets/stage1a-20261021/selection-v3.json
```

运行或中断后继续，使用同一命令；已有进程持有锁时会拒绝第二个写入者：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /data/ticvla/envs/warehouse-io/bin/python -u \
  -m ticvla.warehouse.dataset_tools.r2r_full_prepare run \
  --output /data/ticvla/datasets/r2r-full-20261112 --workers 4
```

入口自动用 `navila-eval` Python 回放 Habitat，用 `warehouse-io` 生成地图、关键帧、schema v2 与教师待标注包；不需要 Isaac Sim。24 条回放组成一个有界分片，剩余磁盘低于40GiB时停止。已完成阶段不重做；进程中断后的部分文件移入 `interrupted-attempts`。明确记录为失败的任务保留失败状态，不自动用其他任务替换。

本次后台启动信息在 `worker-process.json`，输出在 `worker.log` 和 `logs/replay-*.log`。浏览器页面每60秒刷新：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8798 \
  --bind 0.0.0.0 --directory /data/ticvla/datasets/r2r-full-20261112
```

进入 `http://localhost:8798/`。远程机器需转发8798端口；这是本地服务，不是 GitHub 托管页面。

## 产物及未执行项

每个 `episodes/r2r-<ID>/` 包括：

- `capture/raw`：同步 RGB-D、内外参、源位姿、时间及独立任务元数据。
- `verification-only.json`：原始任务/GT/世界原点，仅供核验，不能进入教师输入。
- `preannotation/samples`：当前/历史 RGB、截至源时刻的几何地图和标注辅助。
- `records`、`manifest.json`：schema v2。文本和动作均为 `unlabeled`，几何监督独立保存。
- `schedule.json`：关键帧与参考线覆盖检查；最大时间间隔5秒，方向变化或参考线余量不足时提前更新。
- `teacher-bundle`：按任务排列的完整关键帧。waypoint/ref_line 来自示范轨迹，位于监督字段，不作为教师输入。
- `rgb-replay.mp4`、`index.html`：逐任务视频与关键帧图；不伪造 Landmark/Local/Route 输出。
- `replay-report.json`、`preparation-report.json`、`ready.json`：回放、几何与待标注包的分层状态。

根目录 `status.json` 覆盖全部任务，失败和未执行仍在分母；`manifest.json` 在完整运行或显式子集运行结束后汇总通过几何检查的训练用途记录。`optimizer_updates_permitted=false`。

**本轮几何准备完成不等于语义标注完成。** 检测/OCR 缓存仍标为 `not_prepared`；恢复标注前须用冻结室内词表准备真实感知缓存。语义地图、Previous Route/Local 的递归提交依赖真实教师响应，本轮不能预生成或以人工答案代替。冻结 Plus 配置只作后续配置参照；不会在准备任务结束后自动调用 API。

## 验收

开始全量前，两个真实 Habitat 回放组、6个任务、54个关键帧已通过，回放位置最大误差0m；复用隔离测试2项、坐标与覆盖测试3项通过。该小批结果不代表其余任务通过。

全量以 `status.json` 的 `all_geometry_prepared`、各失败原因及最终 manifest 为准。参考线与锚点一致、没有提前耗尽覆盖、文本/动作缺失监督不贡献 loss；所有语义标签仍待生成和审核。实际回放失败、几何覆盖失败或待标注包构造失败都须报告，不能称为完整成功。
