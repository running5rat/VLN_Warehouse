# 持久语义地图与 LLM I/O（v0.6 阶段实现）

本阶段验收语义地图的工程正确性，以及真实 InternVL 的输入、结构化输出、
参考线/上下文接口。不训练，不评价 LLM 输出的语义或规划质量，不驱动机器人。
原 `warehouse_mvp` 训练/评估入口保持独立。

## 实现与边界

`ticvla.warehouse.mapping` 提供单机器人持久地图：SQLite 保存来源观测、语义事件、
地标、子图与配准约束；不可变 NPZ 保存源深度及三维点。元数据提交之前先完整落盘
数组，重复观测/事件校验内容且不重复增加证据。地图配置和 Oracle/deployment 标记
创建后不可混用；原始 RGB/深度带哈希，文件被修改时拒绝处理。

所有深度障碍参与局部几何检查，当前地图沿用 16 m、0.1 m 分辨率和 5 秒窗口。
持久语义几何采用 0.05 m 体素。mask 边界侵蚀一像素并按深度中位数/MAD剔除
背景离群点，然后使用源标定、源 odom 投影。表面中心不是物体真实几何中心；保存
原始可见表面，以点云重叠复核关联。朝向不可观测时保持 `null`。

人工核验静态性和 ROI 的对象可以进入 verified 层；LLM 自动提示生成的对象先为
tentative，不自动晋升为永久定位锚点。人、车辆等动态类别不进入静态锚点。
同类重复对象的文本提示返回多实例时要求 ROI，不任意选最大置信度实例。
Local/拓扑关系事件保存为明确拒绝结果，当前没有把语言位置转换成 TF 的模块。

同 odom 的新任务可继续关联旧地标。新 odom 创建独立子图，用至少三个空间分散
地标候选做 SE(2) RANSAC，再验证几何重叠及竞争假设。对应不足/歧义时保持分离。
有效约束建立跨子图地标别名；撤销约束会撤销依赖它的子图变换。地标也可单独
撤销并保留原始证据。不是完整 SLAM/位姿图优化或多机器人共享实现。

## 环境与资产

Isaac Sim 在 Ubuntu 22.04 宿主机运行 `/data/isaacsim/python.sh`。训练容器 glibc
2.31 不适合运行此 Isaac 安装。下列离线工具在容器运行，复用现有 tic-vla 环境的
PyTorch/Transformers，新增依赖只安装到独立 venv：

```bash
/root/miniconda3/envs/tic-vla/bin/python -m venv --system-site-packages /data/ticvla/envs/warehouse-io
/data/ticvla/envs/warehouse-io/bin/python -m pip install -r requirements-warehouse-io.txt
export WAREHOUSE_IO_PYTHON=/data/ticvla/envs/warehouse-io/bin/python
export PYTHONPATH="$PWD"
```

依赖固定为 Ultralytics 8.3.203、LM Format Enforcer 0.11.3、OpenCV 4.11.0.86 和
固定提交的 CLIP tokenizer；不向 Isaac Python 安装这些包。

将以下官方发布资产下载到同一模型目录。文本提示使用显式本地 MobileCLIP 文件，
推理不再按当前工作目录自动下载：

```bash
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-v8l-seg.pt \
  -o /data/ticvla/models/yoloe-v8l-seg.pt
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts \
  -o /data/ticvla/models/mobileclip_blt.ts
```

每次真实分割记录版本和资产 SHA256。已有文件无需重复下载。模型和分割器采用
eval/无梯度推理，不执行优化器更新。

## 复现 V0–V3

所有输出使用新目录。`WAREHOUSE_RUN` 必须在共享 `/data` 下，并允许宿主机当前用户
写入。宿主机在仓库根目录、退出 Conda 后运行：

```bash
export WAREHOUSE_RUN=/data/ticvla/outputs/my-mapping-run
PYTHONPATH="$PWD" /data/isaacsim/python.sh tests/warehouse/run_isaac_mapping_capture.py \
  --output "$WAREHOUSE_RUN/capture"
```

采集器生成一个有货架和三个箱体的确定性 USD 场景，记录 13 帧移动相机观测。
实例 ID 单独写入 `raw/supervision`，不进入部署观测。这是相机回放，不是底盘闭环。
文件名从零编号，观测 ID 为 `mapping-a:00000000` 至 `mapping-a:00000012`。

然后在容器、仓库根目录运行：

```bash
export WAREHOUSE_RUN=/data/ticvla/outputs/my-mapping-run
export WAREHOUSE_IO_PYTHON=/data/ticvla/envs/warehouse-io/bin/python
export PYTHONPATH="$PWD"
"$WAREHOUSE_IO_PYTHON" -m unittest discover -s tests/warehouse -v \
  > "$WAREHOUSE_RUN/unit-tests.log" 2>&1
"$WAREHOUSE_IO_PYTHON" tests/warehouse/run_mapping_validation.py \
  --capture "$WAREHOUSE_RUN/capture" \
  --weights /data/ticvla/models/yoloe-v8l-seg.pt \
  --prompts configs/warehouse_mapping_prompts.json \
  --output "$WAREHOUSE_RUN/mapping"
```

提示文件中的 ROI 是人工检查图像后固定的像素区域；`evaluation_prim` 只在评估器
读取真值时使用，不传给 YOLO-E 或地图输入。验收阈值提前固定：mask IoU ≥0.70、
可见表面中心误差 ≤0.15 m、重定位平移误差 ≤0.10 m、yaw误差 ≤0.05 rad。
此外，将投影点与独立的USD箱体解析表面比较，表面距离P95须≤0.10 m；
`evaluation_bounds` 与实例真值一样只由评估器读取。
误差阈值只适用于该固定夹具，不是实机精度承诺。

验证器处理首尾两帧的三个对象，保存真实 mask/叠图；并在独立 `oracle-map` 中用
仿真实例 mask 建对照图。检测或几何失败会令脚本返回非零，并保留报告。
复制地图到 `registration-map` 后进行跨任务、人工 odom 原点变换、拒绝及撤销测试，
不会破坏后续 LLM 所读取的主地图。该重定位测试不是物理搬移后的感知泛化测试。

## 复现 V4–V5

```bash
"$WAREHOUSE_IO_PYTHON" -m ticvla.warehouse.run_io \
  --store "$WAREHOUSE_RUN/mapping/map" \
  --image-id mapping-a:00000012 \
  --task "$WAREHOUSE_RUN/mapping/task.json" \
  --model /data/ticvla/models/InternVL3-1B \
  --vlm-checkpoint /data/ticvla/dataset/TIC-VLA-model.ckpt \
  --segmenter-weights /data/ticvla/models/yoloe-v8l-seg.pt \
  --output "$WAREHOUSE_RUN/io"

"$WAREHOUSE_IO_PYTHON" tests/warehouse/summarize_mapping_io.py \
  --capture "$WAREHOUSE_RUN/capture" --mapping "$WAREHOUSE_RUN/mapping" \
  --io "$WAREHOUSE_RUN/io" --unit-log "$WAREHOUSE_RUN/unit-tests.log" \
  --output "$WAREHOUSE_RUN/validation.json"
```

LLM 输入包含因果历史 RGB（目标偏移 0/1/2/3 秒、使用实际时间/位姿，缺失不伪造）、
已观测 MapView 渲染图、最多16项地标记录、固定任务契约、工作 Route 和机器人状态。
这是 MapView 渲染对照接口，未实现学习型 MapEncoder 或历史 token 压缩。

TEXT 使用 XML 外层、JSON landmark 数组和 waypoint 数字语法的 token 约束；所有
字段内容仍由真实模型选择。类别描述可以错误，格式合法不代表语义正确。语法限制为
最多3项 landmark、80字符 Local、120字符 Route、两位整数/三位小数的 waypoint；
空 landmark 或 waypoint 用 `[]`。完整输出是：

```text
<landmark>[{"image_id":"mapping-a:00000012","category":"box"}]</landmark>
<local>location or unknown</local>
<Route>remaining instruction</Route>
<waypoint>(0,0);(1,0);(2,0);(3,0);(4,0)</waypoint><REFLINE>
```

上面仅为协议示例，实际模型文本保存在 `text/response.json`。推理会记录约束解码
改变贪心选择的次数；不靠替换答案或追加查询标记把截断输出伪装成成功。

TEXT 和 PREFILL_FOLLOW 在同一观测上分别测试，**不是一段跟随任务的连续执行**。
PREFILL 只前向固定查询槽，不生成文本/事件，不改写工作 Route。两种模式均读取
已处理的 `<REFLINE>` 状态，输出参考线 `[32,2]`、前缀 mask `[32]` 和上下文
`[16,512]`。新增头固定随机种子并保存到 `interface-heads.pt`，明确未训练、未验证
几何质量。`snapshot.npz` 与最后提交的 `response.json` 构成完整数据包。

快照时钟为离线源时间加本机测得的处理时长，不能用来声称跨主机同步或在线 age
达标。模型/分割器性能也不是 Orin 测量。输出包含 source/task/map版本和哈希；
LLM 事件提交后另行记录处理结果，重复事件不会重复写图。

## 使用自己的采集数据

原 episode 合约不变，新增的 `capture_oracle` 可选项只控制独立真值导出。

```bash
"$WAREHOUSE_IO_PYTHON" -m ticvla.warehouse.mapping.cli --store /data/my-map import \
  --episode /data/my-episode/episode.json
"$WAREHOUSE_IO_PYTHON" -m ticvla.warehouse.mapping.cli --store /data/my-map segment \
  --events /data/my-events.json --weights /data/ticvla/models/yoloe-v8l-seg.pt
"$WAREHOUSE_IO_PYTHON" -m ticvla.warehouse.mapping.cli --store /data/my-map export \
  --image-id episode-id:00000000 --output /data/my-map-view
```

事件文件是 `SemanticEvent` 列表：`message_id/image_id/category` 必填，
`source=manual|llm`，可选 `roi=[x1,y1,x2,y2]`、`name`、`static_confirmed`。
`static_confirmed` 仅信任已核验人工/Oracle来源，不能用 LLM 自称静态来获得定位资格。
新 odom 使用新子图；`register --source ... --target ...`、`revoke --constraint-id ...`
和 `revoke-landmark --landmark-id ... --reason ...` 提供显式配准与撤销入口。

## 后续差距

已实现：持久地图/来源、真实分割、保守配准、撤销、双模式真实前向、结构化协议与快照。
本轮验证：固定场景地图正确性与 I/O 接线。
未验证/未实现：自动静态性确认、OCR/目标身份、Local几何推理、完整位姿图优化、
学习型MapEncoder、VLM质量与训练、跟随任务、九模式Flow、Scorer、控制/Safety闭环、
实时队列、多机器人和实机部署。具体实测结果见 `warehouse_mapping_io_validation.md`。
