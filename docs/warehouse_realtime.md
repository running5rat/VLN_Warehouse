# 固定词表实时地标感知

## 单帧分阶段延迟测量

`profile_semantic_frame` 在独立地图中对同一源帧执行真实模型推理，区分模型加载、预热、几何、分割、候选投影、请求图生成、VLM、入图、OCR、文字绑定、后处理、绘图发布和归档。VLM 使用 CUDA events 测量视觉编码、首次语言前向（prefill）和后续 token 前向；另记录格式约束回调 CPU 时间。内部计时是嵌套区间，不应与 VLM 总时间重复相加。

```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  /data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.profile_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/instances-original-20261012-154216/capture/raw/episode.json \
  --frame 30 --repeats 5 --array-compression-level 0 \
  --output "/data/ticvla/outputs/frame-latency-$(date +%Y%m%d-%H%M%S)"
```

默认数组压缩等级 0 与实时入口一致；旧单帧入口等级为 6，应避免混合比较。脚本还运行较短 prompt 的诊断对照，保留相同两张图、候选表和输出 schema，不修改生产 prompt。输出包括每次真实回答、图像、地图、阶段计时及汇总 JSON。仅测已预热模型的串行单帧处理，不包含 Isaac 采集、进程间通信、排队或浏览器绘制，也不代表整圈 p95。

2026-10-12 报告：`/data/ticvla/outputs/semantic-frame-latency-20261012-live-storage/index.html`。第 30 帧有两个候选，重复 5 次。当前 VLM 总时间 p50 约 1.47 秒，语言模型 prefill 约 23 ms，decode 前向累计约 1.28 秒。缩短 prompt 后文本与特殊 token 从 403 减为 304，但输出从 61 增为 64 tokens（JSON 空白变化），未观察到整体提速。此次没有部署 prompt 或协议优化。

## 地图后处理视图（2026-10-12）

实时、单帧和回放的 `yoloe-realtime` 显示现已接入可逆的地标视图。原始 SQLite 记录、OCR 证据、几何占用、关联结果和静态 MapView 不变。`mapping/map.png` 显示筛选后的 `keep + review` 对象；`detections.png` / `semantic.png` 继续显示原始检测及成功入图实例，标注 `view:keep/hide/review`。`landmark-view.json` 保存全部对象和过滤原因，包含被隐藏对象。新运行仍使用下文原启动命令。

实现位于 `ticvla/warehouse/mapping/postprocess.py`，由 `MapStore.landmark_view()` 调用。缓存随地标证据变化更新，普通几何帧复用结果。不新增 VLM/OCR 调用。该模块不纠正 YOLO 类别、不补造 OCR 字符，也不赋予静态锚点资格。

身份融合当前采用用户指定的 `landmark-view-v2-substring` 策略：相邻同一行的 OCR 分块可形成规范化候选；原文始终保留。同一实例的合格读数去重后，移除被其他读数包含的严格子字符串；只剩一个父字符串时采用该字符串，不再要求图像边界裁切证据。若剩下两个或更多互不包含的字符串，则 `identity_text=null`，`registration_status=rejected`，放弃该实例的 OCR 标签发布，mask 和几何仍保留。多字符串情况不按长度、出现次数或先后顺序强行选取。只比较连续子字符串，保留标点和大小写；沿用规范化及 0.8 置信度要求。

候选必须来自真实读数；不拼接不同帧碎片。`ignored_substrings` 保存忽略的短读数，`maximal_candidates` 保存剩余的不可包含候选；原始 OCR 是审计证据，不是已采用标签。只有至少三个不同源帧的完整父字符串读数及不同视角支持才能确认身份，子串不增加完整支持次数。阈值和通用文字策略记录在输出 `policy` 中。旧 v1 的裁切规则已经被替代，其历史报告仍保留。

筛选规则：没有文字证据的普通 `storage rack` 默认隐藏；有文字的货架保留为 `review`，避免丢失直接印字载体。标牌保留候选；`CAUTION`、`FRAGILE` 作为显式可配置的通用语义属性隐藏，不做唯一身份。这里的词表是视图策略，不是 OCR 编号模板。`FRAGIL.` 等不确定错读不自动归并。其他设施仍需人工/模型进一步核验，不因类别名而宣称有定位价值。所有决策可由原始记录重算。

仅对已有地图执行后处理，无需启动 Isaac 或模型（输出目录必须是新目录）：

```bash
/data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.run_map_postprocess \
  --map /data/ticvla/outputs/warehouse-loop/instances-original-20261012-154216/mapping/map \
  --output "/data/ticvla/outputs/map-postprocess-$(date +%Y%m%d-%H%M%S)"
```

交付 `index.html`、`before.png`、`after.png`、`comparison.png`、`landmark-view.json`、`changes.json`、`report.json`。图中数字是对照索引，不是实际路标编号。不同子图分开绘制，不叠加未配准坐标。

当前验证目录：`/data/ticvla/outputs/landmark-postprocess-substring-20261012`。最新原始地图仍保留 30 条原始记录，视图隐藏 11 条，显示 19 条。第 9 项采用 `72/4R`、忽略 `/4R`；第 23 项采用 `72/5R`、忽略 `2/5R`。六个编号目标现在均显示完整候选编号，但不因此升级为静态锚点或已确认身份。121 项测试通过；验收是已有真实采集的离线后处理及增量证据回放，没有重跑 OCR 或 Isaac，不是模型准确率或性能提升评测。旧 v1 报告保留在 `/data/ticvla/outputs/landmark-postprocess-20261012/verified`。

实现 `yoloe-realtime`：几何 5 Hz、固定类别实例 mask 5 Hz、InternVL 类别选择每 2 秒一次、共享全文检测和批量 OCR 每秒一次。mask 入图不等待读字；文字通过父实例事件追加，不增加几何观测或独立视角数量。所有地标保持 tentative。

词表为 sign、label、storage rack、door、pillar、fire extinguisher、electrical cabinet。模型检测整个词表，VLM 选择登记子集，词表外建议保留为待扩展。合法空选择清空登记子集；失败结果不补问，最后合法选择按源时间 5 秒过期。类别选择来源与检测图像来源分别保存。JSON 格式约束仅约束一次问答的语法，不指定模型应选择哪些类别。

新增文字事件由 `MapStore.attach_text_evidence()` 消费。它核验父事件、源图与深度哈希、文字区域和状态；填充同帧空文字证据，保留不同读数，重复事件不增加版本或证据。三个不同源帧与两个视角的规则保持不变；身份证据不是静态性证明。

## 环境与启动

Isaac 仍使用宿主 `/data/isaacsim`。新增 `/data/ticvla/envs/warehouse-realtime`，继承现有 ML 包而不升级旧环境；TensorRT 安装在新环境。OCR 固定 PaddleOCR 3.3.2／Paddle 3.2.2 的 PP-OCRv5 mobile det/rec，CPU 4 线程，MKL-DNN 开启，检测长边最大 960，识别批量 16。不使用牌子颜色、长宽比或编号模板。

选型通过后生成 `/data/ticvla/models/warehouse-realtime/selected.json`，包含固定类别引擎路径和来源。没有选中模型时拒绝启动，不回退到 Florence 或旧 OCR。

```bash
bash DynaNav/warehouse_loop/run_realtime.sh \
  --output "/data/ticvla/outputs/warehouse-loop/realtime-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

综合场景增加 `--landmark-variant combined`。物理步进 60 Hz、渲染 5 Hz、RGB-D 5 Hz；车速与轨迹不变。也可在原 `run_live.sh` 命令中指定 `--perception-backend yoloe-realtime`，脚本自动选择新的推理环境。网页绘制频率与地图发布频率分别看待。

单帧：

```bash
/data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.run_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --llm-mode objects-only --base-model-only \
  --perception-backend yoloe-realtime --output /data/ticvla/outputs/realtime-frame-new
```

回放通过 `run_perception_replay` 指定同一 episode/frame、`--response` 指向本模式保存的 response.json。核验 RGB、深度和采集标定／位姿文件哈希后复用类别回答，LLM 调用数为零。旧模式回答不能冒充本轮词表选择。

## 实验与门槛

实验目录 `/data/ticvla/outputs/landmark-realtime-20261012`。`perception.realtime_benchmark` 支持 qa、ocr、infer、score、freeze，开发 12 帧和留出 144 帧来自前轮固定 manifest，不重新筛选成功帧。所有分割候选共享同一真实 VLM 输出和源 OCR 输出，各自使用独立实例绑定与地图。开发阶段结束后冻结模型、词表、提示和消费者代码，留出测试不调整配置。

TensorRT 使用固定 640、batch 1、FP16，不接受运行时新类别。导出在隔离环境显式准备依赖，关闭 Ultralytics 自动安装。导出命令：

```bash
YOLO_AUTOINSTALL=false /data/ticvla/envs/warehouse-realtime/bin/python \
  -m ticvla.warehouse.perception.export_realtime \
  --root /data/ticvla/outputs/landmark-realtime-20261012
```

实时验证使用运动时段的实际完成频率，mask 发布 p95 不超过 500 ms；OCR 从源帧提交到文字发布 p95 不超过 2 秒、完成率至少 95%。被替换、超时、空选择和拒绝均保存，不能靠重复显示或停车后排空队列补算通过。

质量统计仅针对独立标注目标；不能把有文字但无正确归属的实例算成成功。原始、换色、直接印字三组需要真实正确三维文字正例，报告误绑定和误合并。固定词表未包含所有容器类别，因此词表外载体漏检必须作为范围限制保留，不能用人工类别替换 VLM 答案。

## 2026-10-12 实测结果

**实现可运行，整体验收未通过。不能作为已经满足实时、可靠编号建图的版本。**

部署选型为 YOLOE-v8s TensorRT FP16。开发集 12 帧：11s/v8s/v8l 的标注目标 mask 召回分别为 6/36、8/36、8/36，分割 p95 分别为 18.5、19.8、20.3 ms。部署选择遵循开发集规则，留出结果没有用于改选模型或改词表。

| 优化后真实绕环 | RGB-D/几何/分割目标 | 几何实际 Hz | 分割实际 Hz | mask 发布 p95 | OCR 绑定 p95 | OCR 完成 |
|---|---|---:|---:|---:|---:|---:|
| 原始 | 5 Hz | 4.86 | 4.83 | 3.46 s | 3.78 s | 91/91 |
| 综合变化 | 5 Hz | 4.84 | 4.81 | 4.05 s | 4.25 s | 91/91 |

每圈 RGB-D 448 帧，几何全部消费。原始场景替换了 1 个分割候选，综合场景没有替换；没有依靠大量丢帧维持频率。原始/综合场景运动期间分别有 6/3 次文字属性更新，但次数不代表正确编号：原始帧 275 的 `/5R` 是正确绑定的部分文字，完整 `72/5R` 未读出；综合场景观察到箱体警告词错误归属货架。两圈延迟均超过 500 ms/2 s 门槛。

留出集 144 帧、385 个已标注目标、133 个可读唯一编号目标：

| 后端 | mask IoU≥0.5 的检出 | 正确编号且已入图的 3D 观测 | 错误文字归属 |
|---|---:|---:|---:|
| 11s TensorRT | 66/385 | 2/133 | 0 |
| v8s TensorRT | 96/385 | 3/133 | 2 |
| v8l TensorRT | 90/385 | 3/133 | 1 |

三个模型共用本轮 144 个真实 VLM 回答和 CPU mobile OCR 结果。VLM 选择覆盖 53/385 个标注目标；其余设施未完整标注，不能把此比例作为所有物体的选择准确率。v8s 的严格文字匹配为 53/133，包含检出但未被 VLM 选择的实例；最终正确三维编号仅 3/133。分割成功样本平均 mask IoU 为 0.896，**这是条件平均值，不是全数据平均质量**。正确三维编号统计允许拼接 OCR 分开的文字区域，严格字符串匹配另报。

换色与直接印字两组均没有符合端到端条件的编号正例。确认的错绑样本是 `surface-00000010`：`CAUTION` 与 `K4-73` 来自两个容器，但被覆盖它们的 storage rack mask 吸收。没有观察到已标注目标的错误合并；所有方案均未产生 confirmed identity，不能据此宣称身份核验可靠。地图中的设施候选全部保持 tentative。

固定第 200 帧的单帧和回放入口完成真实模型调用，但 VLM 选择 storage rack/door/pillar，YOLO-E 检出 fire extinguisher，类别交集为空，因此没有物体入图。这是失败的端到端回归样例，未用人工提示替换。

98 项回归测试通过；真实两圈地图重启后，来源观测、实例事件和文字事件重放均未增加地图版本或几何证据。体素优化用全部 156 帧缓存 mask 重放，三维点数组与入图状态逐项完全一致。

## 运行工件与复现

主目录：`/data/ticvla/outputs/landmark-realtime-20261012`。

- `index.html`：144 帧并排分割与文字叠图、分场景 BEV/3D 地图。
- `acceptance.json`、`quality-report.json`：总体失败项、逐场景分母与计分范围。
- `live-original-optimized/`、`live-combined-optimized/`：两圈 RGB-D、事件、地图、队列日志、GPU 采样和实时报告。
- `manual-review.json`：正例、错绑及运动中文字更新的目视核验范围。
- `single-frame-200/`、`replay-frame-200/`：固定回归帧原文与失败结果。
- `environment-lock.txt`、`model-lock.json`、`engines/*/model.engine.json`：依赖、权重哈希、词表、TensorRT 版本。
- `freeze.json`：开发集选型冻结；`execution-freeze.json`：保持同一推理配置的等价 CPU/写盘优化；`optimization-parity-v8s-engine.json`：156 帧一致性证据。
- `live-original/` 保留优化前的失败圈。问答格式调试、旧 Transformers 兼容失败及未缓存索引的尝试也分别保留，没有覆盖失败记录。

新环境的部署依赖固定 Transformers 4.57.6、Tokenizers 0.22.2，与原 InternVL 环境一致；旧感知环境中的 Transformers 4.49.0 不兼容该约束解码首步。`setup_realtime_env.sh` 只安装到新环境。tokenizer 约束索引在预热中构建并缓存，12 帧开发集缓存前后原文完全相同，没有新增一次问答。

```bash
# 当前环境已装好；需要恢复新环境时才执行
bash DynaNav/warehouse_loop/setup_realtime_env.sh

# 原始场景；启动前完成模型加载和预热
bash DynaNav/warehouse_loop/run_realtime.sh \
  --landmark-variant original \
  --output "/data/ticvla/outputs/warehouse-loop/realtime-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer

# 查看已生成的实验结果，不运行模型或 Isaac
/data/ticvla/envs/warehouse-realtime/bin/python -m http.server 8766 \
  --directory /data/ticvla/outputs/landmark-realtime-20261012
# 浏览器访问 http://localhost:8766/

# 回归测试
/data/ticvla/envs/warehouse-realtime/bin/python -m unittest discover -s tests/warehouse -q
```

代码入口：`perception/realtime.py` 实现选择、固定词表分割、全文 OCR 和共享消费者；`mapping/store.py::attach_text_evidence()` 实现迟到文字事件；`warehouse_loop/realtime_mapping.py` 负责独立模型队列和单一地图写入者；`realtime_benchmark.py`、`realtime_report.py`、`realtime_acceptance.py` 负责可追溯评测。

尚未解决的主要差距是地图关联与图像发布的 CPU 突发积压，以及大设施 mask 覆盖未独立分割的容器时无法证明真实文字承载对象。下一步应把投影准备和显示发布进一步移出地图写入路径，并在独立开发集解决承载实例漏检/保守拒绝；不能单纯降低验收帧率、提高允许延迟或把候选读数改名为已确认编号。

完整离线选型/质量重跑使用新目录，沿用独立采集与标注：

```bash
export YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
RT_PY=/data/ticvla/envs/warehouse-realtime/bin/python
RT_TEST_ROOT="/data/ticvla/outputs/realtime-benchmark-$(date +%Y%m%d-%H%M%S)"
"$RT_PY" -m ticvla.warehouse.perception.export_realtime --root "$RT_TEST_ROOT"
for RT_STAGE in qa ocr; do
  "$RT_PY" -m ticvla.warehouse.perception.realtime_benchmark --root "$RT_TEST_ROOT" --stage "$RT_STAGE"
done
for RT_MODEL in 11s v8s v8l; do
  for RT_RUNTIME in pt engine; do
    for RT_STAGE in infer score; do
      "$RT_PY" -m ticvla.warehouse.perception.realtime_benchmark --root "$RT_TEST_ROOT" \
        --model "$RT_MODEL" --runtime "$RT_RUNTIME" --stage "$RT_STAGE"
    done
  done
done
"$RT_PY" -m ticvla.warehouse.perception.realtime_benchmark --root "$RT_TEST_ROOT" --stage freeze
for RT_STAGE in qa ocr; do
  "$RT_PY" -m ticvla.warehouse.perception.realtime_benchmark --root "$RT_TEST_ROOT" --split evaluation --stage "$RT_STAGE"
done
for RT_MODEL in 11s v8s v8l; do
  for RT_STAGE in infer score; do
    "$RT_PY" -m ticvla.warehouse.perception.realtime_benchmark --root "$RT_TEST_ROOT" \
      --split evaluation --model "$RT_MODEL" --stage "$RT_STAGE"
  done
done
"$RT_PY" -m ticvla.warehouse.perception.realtime_report --root "$RT_TEST_ROOT"
```

`freeze` 会更新本机 `selected.json` 部署选择，因此质量重跑后的实时启动使用该次开发集选型。总体 `acceptance.json` 还需要两圈实测、恢复检查与独立目视核验记录；单独完成离线脚本不代表总体通过。

## 候选实例筛选模式（2026-10-12）

`yoloe-realtime` 新增 `--selection-mode instances|categories`，默认 `instances`。
实例模式先检测，再用一次 InternVL 问答选择源帧实例 ID；输入原始 RGB、带 ID 的框叠图和候选表，不输入导航任务、OCR 读数或评测真值。车辆继续沿固定轨迹运动。`categories` 保留旧的类别选择行为。

实例选择不跨帧传播。检测和候选投影为 5 Hz 目标，VLM 每 2 秒采样，OCR 每秒采样；VLM 源帧与 OCR 对齐。正式入图必须等待该源帧的选择，不能把候选频率算成正式入图频率。待决源帧最多保留 5 秒、32 帧；过期或替换会记录，不用当前位姿处理历史结果。OCR 先到时等待对应实例准入，后到时只追加文字证据。所有新对象仍为 tentative。

```bash
# 一个脚本启动；默认打开 Isaac GUI，浏览器入口打印在终端
bash DynaNav/warehouse_loop/run_realtime.sh \
  --selection-mode instances --landmark-variant original \
  --output "/data/ticvla/outputs/warehouse-loop/instances-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer

# 旧类别模式
# 将上面的 --selection-mode instances 改为 --selection-mode categories

# 固定第 200 帧；无需 Isaac
PYTHONPATH="$PWD" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
/data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.run_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --perception-backend yoloe-realtime --selection-mode instances \
  --llm-mode objects-only --base-model-only \
  --output /data/ticvla/outputs/instance-frame-new
```

三组回归：A 复用原问答并精确匹配类别，B 只放通 sign/label 登记范围，C 用建图实例问答。三组共用原有真实分割和 OCR 结果，地图各自独立；离线缓存读取时间不算模型速度。C 还改变了图像输入和问题，因此是整体方案对照，不是单一因素消融。原有 144 帧已被分析，只称回归集。

```bash
export PYTHONPATH="$PWD"
INSTANCE_PY=/data/ticvla/envs/warehouse-realtime/bin/python
INSTANCE_ROOT="/data/ticvla/outputs/instance-comparison-$(date +%Y%m%d-%H%M%S)"
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_benchmark --root "$INSTANCE_ROOT" --stage qa --split development
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_benchmark --root "$INSTANCE_ROOT" --stage maps --split development
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_benchmark --root "$INSTANCE_ROOT" --stage freeze
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_benchmark --root "$INSTANCE_ROOT" --stage qa --split evaluation
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_benchmark --root "$INSTANCE_ROOT" --stage maps --split evaluation
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_report --root "$INSTANCE_ROOT"
```

主实验目录 `/data/ticvla/outputs/landmark-instances-20261012`。首轮空候选约束问题保留在同级 `landmark-instances-20261012-attempt-01`；安装的约束库未正确执行 `maxItems=0`，现为空候选使用只允许空选择列表的语法约束，仍执行真实问答并由模型生成漏检类别。无人工替代答案。

离线 sign 的分割检出三组均为 80/320。116 个可读 sign 编号观测中，正确三维入图 A/B/C 为 3/13/49；直接印字容器另计 0/6/11（共 17 个可读观测），不代表固定词表学会了 container 分类。新 VLM 在 117 个有候选帧中的 115 帧全选，因此恢复了准入召回，但尚未实现可靠的去冗余或误检纠正。三组均观察到 2 次入图文字错绑；不得将增加入图数量等同于准确率全面改善。

实例模式两圈实测：

| 场景 | 几何 Hz | 候选完成 Hz | 候选发布 p95 | 正式入图帧 Hz | 文字绑定 p95 | 分割待处理替换 |
|---|---:|---:|---:|---:|---:|---:|
| 原始 | 4.99 | 4.31 | 0.559 s | 0.324 | 3.257 s | 60/448 |
| 综合变化 | 4.98 | 4.18 | 0.625 s | 0.357 | 3.921 s | 72/448 |

两圈约 51.59 m，均消费 448/448 几何帧。106 项测试通过，离线及两圈实际地图重启后重放实例与文字事件不增加版本或证据。运动中核实到完整编号正例，详见 `manual-review.json`；实时目标仍未通过，不能依赖被替换的帧宣称 5 Hz。正式入图频率仅计至少一个实例 accepted 的帧，空选择仅计问答完成。计数修正依据原始日志重算，见 `reporting-correction.json`，没有重新生成模型结果。

`index.html` 提供 144 帧及 385 个标注目标的 A/B/C 对照图，`analysis.html` 给出失败图片和原因。典型剩余失败：`color-00000007` 的 K4-72 已被 OCR 正确读取但归属歧义；`original-00000057` 的 72/5R 只读出 /5R；`layout-00000057` 将 K5-71 读成 Ki-71；`surface-00000010` 将容器文字挂到货架。第 200 帧的新模式选择了三个被误标为灭火器的候选，不能作为正确标号地标的成功样例。

```bash
# 使用同一个 INSTANCE_ROOT 保存两圈实际模型运行（本轮实测为 headless）
bash DynaNav/warehouse_loop/run_realtime.sh --selection-mode instances \
  --landmark-variant original --headless --port 8781 --output "$INSTANCE_ROOT/live-original"
bash DynaNav/warehouse_loop/run_realtime.sh --selection-mode instances \
  --landmark-variant combined --headless --port 8782 --output "$INSTANCE_ROOT/live-combined"
# 完成两圈后汇总验收报告；回放去重检查会打开本次地图并验证状态不变
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_recovery_check --root "$INSTANCE_ROOT"
"$INSTANCE_PY" -m ticvla.warehouse.perception.instance_delivery --root "$INSTANCE_ROOT"

# 浏览器打开 http://127.0.0.1:8780/analysis.html
/data/ticvla/envs/warehouse-realtime/bin/python -m http.server 8780 \
  --bind 127.0.0.1 --directory /data/ticvla/outputs/landmark-instances-20261012
```
