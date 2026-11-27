# Stage1a：按顺序验收训练、数据与教师标注

当前 schema v2 保持不变。执行顺序是 A 人工训练诊断、B 几何与待标注数据、C 教师比较与审核、D 批量标签和正式训练。每阶段报告通过后再进入下一阶段；API 配置只在 C 阶段需要。

**2026-10-28 更新：**最新设计以 [VLM 标注与输入输出规范](vlm_annotation_and_io_spec.md) 为准。当前只继续比较 Qwen，GPT pending，GLM 等历史结果保留；下文五模型方案是早期计划，不是自动恢复调用的指令。证据输入已精简，详细教师 SYSTEM 与短学生 SYSTEM 尚待分离；新模板接入后必须重新验证训练/部署一致性及自由生成，旧 A 阶段结果不能代替新模板验收。200 个 episode 的几何准备不代表已补齐新语义地图输入或已批准语义标签。动态 Oracle、FMAE、Scorer 暂缓。

## A：人工数据的真实训练诊断

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.train_smoke \
  --project /data/ticvla/datasets/manual-static-004/annotation \
  --output /data/ticvla/outputs/stage1a/manual-smoke-new
```

基础模型默认 `/data/ticvla/models/InternVL3-1B`，用 `--model` 覆盖。命令拒绝覆盖输出目录。只更新语言 attention 的 LoRA 和 Reference Decoder，100 个 optimizer steps，每步积累 4 个单样本。BF16 基础模型、FP32 可训练参数；LoRA 学习率 3e-5，Decoder 1e-4。使用 SDPA 与非 reentrant 梯度检查点，不删减历史图像或监督文本来节省显存。

TIC-VLA 初始化兼容检查独立运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.train_smoke \
  --project /data/ticvla/datasets/manual-static-004/annotation \
  --legacy /data/ticvla/dataset/TIC-VLA-model.ckpt --steps 1 \
  --output /data/ticvla/outputs/stage1a/tic-compat-new
```

人工导出仍是禁止 optimizer 更新的 regression manifest。入口另外创建限定步数的 diagnostic manifest，保留用户明确授权、源文件哈希及 `training_examples_only` 评估范围。正式训练不能使用这两种 manifest。普通 schema v2 训练 manifest 需要显式 train/val/test 与来源分组，并保留记录哈希；同一 episode 不能跨 split。

输出包含：

- `export/`：原 schema v2 回归导出；`manifest.json`：本次有限训练清单。
- `check-0000/`、`check-0020/`、`check-0100/`：实际输入、生成文本、严格格式解析结果、teacher-forced 与生成后 Decoder 数组及参考线对照图。
- `metrics.jsonl`、`loss.png`：每次参数更新的分项 loss、梯度范数、耗时与峰值显存。
- `last.pt`：诊断 checkpoint，不能作为正式实验初始化或“已训练导航模型”。
- `source-hashes-before.json`、`source-hashes-after.json`：所核验的原始文件范围与哈希。
- `report.json`、`index.html`：训练链路、损失变化、冻结边界及部署重载检查。

全部样本来自同一人工 episode，训练后同样本结果只表示拟合；格式合法率和自由生成参考线质量分别记录。基础模型未自行生成 `<REFLINE>` 时，额外处理的查询只用于诊断，输出明确标记无有效完整生成。

缺失文本监督仅处理 `<REFLINE>` 查询，不构造假 Landmark/Local/Route，也不把未来 waypoint 作为输入。缺失动作保留全 false mask；`valid_empty` 参考线可监督有效长度为零，与完全未标注不同。

## 后续阶段边界

B 阶段首批准备 R2R 120、英语 RxR 80 个 episode，理想 odom + 同步 RGB-D，参考线和锚点来自回放轨迹。语义标签保持未标注。独立保留教师比较样本，不修改官方验证集。

C 阶段在用户配置 API 后，实测 GPT-6 Luna、GPT-6.1 Sol、GPT-6 Astra、GLM 最新可用视觉模型和 Qwen 最新可用视觉模型。具体模型、版本、端点在调用前核验并冻结；未调用不能宣称教师验证通过。未来几何只用于标签，不提供给语义教师。

D 阶段须等待教师比较及审核通过。动态 Oracle、FMAE、Scorer 本轮不运行。

## B：原始数据配置与复现

本地 NaVILA 的 RxR 评测配置改成了 15°、允许滑动，不能用于回放原始 RxR GT。回放使用原始 VLN-CE 配置：R2R 为 15°、半径 0.10 m、允许滑动；RxR 为 30°、半径 0.18 m、高度 0.88 m、禁止滑动。Habitat-Sim 根据机器人尺寸重算导航网格，仅在内存中使用，不覆盖原场景文件。原始配置来源：<https://raw.githubusercontent.com/jacobkrantz/VLN-CE/master/habitat_extensions/config/rxr_vlnce_english_task.yaml>。

R2R 同步 RGB-D 为 512×512、90°、相机高 1.25 m；RxR 为 640×480、79°、相机高 0.88 m。两者都由同帧相机获取深度，保留米制光轴 Z；训练预处理再按现有 448 接口处理。初始转向与楼梯高度变化保留在原始位姿中，XY 参考线不代表完整差速车动力学监督。

```bash
# 固定 200 个训练任务与 8 个独立教师比较任务。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.stage1a_data \
  catalog --output /data/ticvla/datasets/stage1a-new

# 使用已安装 Habitat-Sim 的环境；无需 Isaac Sim，也不加载 VLM。
PYTHONPATH="$PWD" /root/miniconda3/envs/navila-eval/bin/python \
  -m ticvla.warehouse.dataset_tools.stage1a_data replay \
  --selection /data/ticvla/datasets/stage1a-new/selection.json \
  --output /data/ticvla/datasets/stage1a-new/episodes

# 几何监督、历史地图、关键帧、视频和 schema v2 清单。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.stage1a_data \
  prepare --selection /data/ticvla/datasets/stage1a-new/selection.json \
  --output /data/ticvla/datasets/stage1a-new/episodes
```

每个任务保存原始 RGB-D、动作与源位姿、独立 `verification-only.json`、`replay-report.json`、`schedule.json`、`records/`、`manifest.json`、`rgb-replay.mp4` 和关键帧图。世界目标和 GT 路径仅留在核验侧；训练记录没有这些字段。原数据中缺失的单词时间戳只在核验 JSON 中记为 null 并列出字段路径，原始数据不变；它们不被误用为动作回放时间。

`--limit 2` 可先跑两个固定试点。失败任务不另选成功样本替代。回放逐个 MOVE_FORWARD 对比原始 locations，最大误差必须不超过 5 cm；失败回放不能导出训练记录。所有派生文本保持未标注，manifest 明确禁止直接启动 optimizer。`report.json` 含完整失败分母，不能只按成功导出记录宣布整批通过。

## C：五个 Oracle 比较准备与凭据边界

当前比较包位于 `/data/ticvla/outputs/stage1a/oracle-comparison-20261021`，每模型 74 个时刻，共五模型。10 个来自人工 demo；32 个用于模型自己的递归 Route/Local 更新；另 32 个共同历史上下文尚未人工核验，包中标记为 `context_ready=false`，不会自动执行该组。不能把原始 instruction 每轮重复输入称为 gold previous Route。

OpenAI 使用 Responses；GLM、Qwen 使用 Chat Completions，保持相同因果文本和图像内容，最后经过相同部署协议校验。实际输出中的 `reasoning_content` 等保留在原始响应里，不作为监督标签。截断、拒绝、错来源和解析失败均保存，不自动二次修复。几何标签只在响应组装阶段加入。

```bash
# 新建比较包（原包已准备好，无需重复）。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.oracle_comparison prepare \
  --manual-export /data/ticvla/outputs/stage1a/manual-smoke-20261021-b/export \
  --dataset /data/ticvla/datasets/stage1a-20261021/episodes-v4 \
  --output /data/ticvla/outputs/stage1a/oracle-comparison-new

# 无 API 调用的输入检查；删除 --prepare-only 后才会尝试已配置的端点。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.oracle_comparison run \
  --pack /data/ticvla/outputs/stage1a/oracle-comparison-20261021 \
  --config /data/ticvla/outputs/stage1a/oracle-comparison-20261021/configs/gpt-6.1-sol.json \
  --output /data/ticvla/outputs/stage1a/oracle-preview-new --prepare-only
```

模型配置只保存密钥的环境变量名。OpenAI 默认 `OPENAI_API_KEY`，GLM 默认 `ZAI_API_KEY`，Qwen 默认 `DASHSCOPE_API_KEY`。Qwen 端点留空，需要先确认区域和端点；配置中的价格目前按新加坡官方价格估算，不把它自动套用到其他区域。

提供凭据文件后，可用 `--credentials-file /home/xinglinlu/.config/llm/credentials.env`。工具只按 dotenv 数据解析，不执行 `source`；容器中该绝对路径不存在时，尝试同一路径的 `/proc/1/root` 宿主机映射。只加载四个提供商前缀的 `API_KEY`、`BASE_URL`，不保存密钥。实际请求仍使用冻结配置中的 `base_url`、`api_key_env`，不会根据密钥猜测服务区域。

首次执行可加 `--max-bundles 1`，先检查一组真实请求；后续对同一输出目录使用 `--resume`。恢复要求配置与比较包哈希相同，已完成的组只读取保存结果，不重复付费；中途崩溃的组需要先核验，不自动重试。配置中 `stop_on_transport_error=true` 时，HTTP、网络或预算错误会停止该模型的后续提交，格式失败仍保留在比较分母中。

五个配置共享一个 100 美元预算账本，HTTP 前按保守 token 预算预留费用；未知结果保留预留额，重复请求拒绝自动再次付费。返回 usage 后按官方单价估算费用，标记为未计优惠的估算而非账单；超出预留的异常用量会阻止后续调用。每个模型的失败、未执行和待审核样本仍保留在固定分母中。

实际 API 可用性和标注质量尚未验收，D 阶段未启动。共同历史上下文须先形成真实人工审核记录再冻结，随后完整比较；教师输出还需逐字段人工审核，模型自己一致不代表正确。

## 本次实际阶段结果（2026-10-21）

- A：100 步基础模型训练通过；平均文本 loss 1.12594 → 0.37756，归一化参考线点 Smooth-L1 loss 0.04425 → 0.00438。这不是米制位置误差。
- 训练样本自由生成完整格式通过数为 0/10 → 5/10；不宣称泛化或导航成功。
- TIC-VLA 初始化：一次 optimizer 更新、冻结检查和严格重载通过。
- B：固定 208 个 episode 全部回放核验通过；200 个训练 episode 对应 3,458 个待语义标注样本；8 个比较 episode 有 145 个候选时刻，从中固定抽取 64 个。
- R2R、RxR 几何样本各完成真实前向，文本 loss 为 0，未进行 optimizer 更新。
- C：比较包、输入预览、两种 API 适配和预算检查已准备，实际 API 调用 0 次，等待凭据及共同上下文审核。

总览：`/data/ticvla/outputs/stage1a/index.html`。可执行 `python -m http.server 8000 --directory /data/ticvla`，浏览 `/outputs/stage1a/index.html`；跨机器访问时将 localhost 改为服务器地址，或用现有 SSH 端口转发。

## C 阶段真实接口验证（2026-10-22）

凭据位于宿主机 `/home/xinglinlu/.config/llm/credentials.env`，未复制到仓库。用户确认 GLM 使用智谱国内站、DashScope 使用北京区，实际冻结配置位于 `/data/ticvla/outputs/stage1a/oracle-live-20261022-v2/configs/`。

- OpenAI：模型列表查询通过；首次图像请求发现 JSON mode 要求用户消息显式出现 JSON，已修正封装并保留失败原文。修正后的请求返回 `credit_balance_exhausted`，停止后续 OpenAI 请求；中断请求保留 outcome unknown 与费用预留，不当作零费用或成功。
- GLM：`https://open.bigmodel.cn/api/paas/v4`，密钥变量 `GLM_API_KEY`。
- Qwen：`https://dashscope.aliyuncs.com/compatible-mode/v1`，密钥变量 `DASHSCOPE_API_KEY`。
- RGB 保留原分辨率，统一 JPEG quality=95、4:4:4 编码，减少未压缩 PNG 上传量；MapView 不重编码。记录原文件及传输字节的哈希，原采集文件不变。该编码配置同时用于所有 v2 比较请求。
- 国内模型按人民币原价记录费用估算；GLM 输入/输出为 2/7 元每百万 token，Qwen 为 12/36 元每百万 token。共享美元预算保守按每 1 元预留 1 美元，此比例仅为预算上限策略，不能当作汇率或实际美元成本。来源：[GLM 官方价格](https://docs.bigmodel.cn/cn/guide/start/pricing)、[Qwen 官方价格](https://help.aliyun.com/zh/model-studio/model-pricing)。

```bash
# 恢复同一配置和比较包：已完成组不再请求；待人工审核的 gold 组继续跳过。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.oracle_comparison run \
  --pack /data/ticvla/outputs/stage1a/oracle-comparison-20261021 \
  --config /data/ticvla/outputs/stage1a/oracle-live-20261022-v2/configs/glm-5.3-flashx.json \
  --output /data/ticvla/outputs/stage1a/oracle-live-20261022-v2/glm-5.3-flashx \
  --credentials-file /home/xinglinlu/.config/llm/credentials.env --resume

# 只从已保存结果更新总览，无网络请求。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.oracle_comparison summarize \
  --pack /data/ticvla/outputs/stage1a/oracle-comparison-20261021 \
  --output /data/ticvla/outputs/stage1a/oracle-live-20261022-v2
```

结果以该目录 `comparison-report.json`、逐组 `report.json` 及 `index.html` 为准。首次传输失败目录 `oracle-live-20261022` 不覆盖；所有尝试共用原比较包的预算账本。C 阶段仍需补齐 OpenAI、共同上下文审核和逐字段人工审核；不据部分格式成功率选择赢家或启动 D 阶段。

本轮已完成两家国内服务的可执行部分，固定分母均为 42 个时刻（人工 10、R2R 16、RxR 16），另外各有 32 个共同上下文待审核：

| 模型 | 实际请求/响应 | 完整格式通过 | 现有规则通过 | 请求端到端 p50 / p95 | 用量费用估算 |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3-FlashX | 33 / 33 | 26 | 25 | 10.65 / 29.97 秒 | ¥0.696 |
| Qwen3.8-Max | 42 / 42 | 42 | 42 | 52.00 / 116.16 秒 | ¥6.595 |

GLM 的 9 个时刻因前一轮失败而未执行，仍留在 42 的分母中；7 个格式失败主要为 Local 超过 80 字符或含当前语法不支持的 `~`，没有裁剪或重问。人工第 1703 帧的 `completed` 缺少要求的到达事实，另有一条 `wait` 与后续约 2 m 示范参考线的差异被列为人工复核项。Qwen 的全部自动检查通过不代表语义准确率为 100%。费用为官方原价乘实际用量，未计缓存或赠送优惠，不是账单。

共保存 75 条真实生成响应；包含 OpenAI 早期错误及中断在内提交了 79 次请求。OpenAI 两条中断结果仍未知，相关预留不释放为虚构的零费用。33 组双方共有输入的图像及因果上下文核对通过（递归历史按设计各自生成）；75 份教师输入没有监督字段泄漏，186 个源文件哈希不变。22 项相关测试通过。

全部教师提议仍未人工批准；完整五模型对照未完成，Stage D 未启动。回归基准的字段一致性、自动协议检查、真正的语义正确性分别报告。离线复算对照与检查、不调用 API：

```bash
PYTHONPATH="$PWD" /data/ticvla/envs/warehouse-io/bin/python \
  /data/ticvla/outputs/stage1a/oracle-live-20261022-v2/refresh_review.py
```
