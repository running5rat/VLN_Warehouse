# R2R 语义标注试点

本轮固定 `r2r-5351 / r2r-6244 / r2r-6508 / r2r-7275` 四个完整任务，共 40 个预定关键帧。它们来自已有 R2R-VLN-CE 训练数据中的教师比较组，保留比较/回归用途，不自动并入训练池，也不是新的独立盲测。

数据使用 Habitat 中已核验的示范动作回放。RGB-D 同步，`T_odom_body` 由仿真真实位姿相对任务起点转换；这是**理想局部 odom**，不是轮速/IMU/VIO 测量。时钟为每动作 0.2 秒的合成回放时间。无需启动 Isaac Sim，也不进行真实车辆控制或正式训练。

## 固定模型与输入

- 教师沿用 [Plus 冻结配置](qwen_plus_annotation_profile.md)：`qwen3.7-plus`、思考预算 2048、输出上限 8192、北京端点；详细提示和输入格式不另改。
- 检测使用本地 YOLOE-v8s + MobileCLIP、PyTorch FP16；原图 mask、640 输入、conf 0.25、max_det 100，缓存类别特征。不能使用仓库词表 TensorRT 引擎冒充室内配置。
- 词表见 [r2r_indoor.v1.json](../configs/dataset_preparation/r2r_indoor.v1.json)，26 类，独立类别/别名/哈希；没有 `shelf → storage rack` 等仓库映射。默认仓库词表保持兼容。
- OCR 保留 PP-OCRv5 mobile CPU。没有文字的物体同样可以入图，所有对象保持 tentative。
- 当前/历史 RGB、当前检测框叠图、之前提交的语义地图、原 instruction、最近有效 Route/Local 和相对运动共同构成输入。真实像素 mask 用于投影，另作审核图，不冒充实际 VLM 输入。
- 教师只输出 `landmark / local / route`。waypoint/ref_line 使用原始回放轨迹 GT，仅进入监督与审核侧。没有未来图像、隐藏目标坐标或人工语义答案输入。

## 复现流程

从仓库根目录运行。以下为**新实验目录**示例，已有结果不要覆盖；相同付费请求受共享账本去重保护。先检查感知结果，再运行教师。

```bash
# 1. 冻结40帧试点及独立开发集12帧；校验原始数据、轨迹覆盖。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot initialize \
  --dataset /data/ticvla/datasets/stage1a-20261021/episodes-v4 \
  --output /data/ticvla/outputs/stage1a/r2r-pilot-new

# 2. 感知必须使用已装 PaddleOCR 的 warehouse-realtime 环境。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DISABLE_MODEL_SOURCE_CHECK=True \
/data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot prepare \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new --development

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot precheck \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new

# 查看 development-0.jpg、development-1.jpg 和 development-review.json。
# 该检查只验证接口；可另存视觉复核结论，不能自动宣称检测正确。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DISABLE_MODEL_SOURCE_CHECK=True \
/data/ticvla/envs/warehouse-realtime/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot prepare \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new

# 3. 每个任务单独调用；严格按此顺序：5351、6244、6508、7275。
# 后续任务会检查之前任务的请求完整性与因果审计。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot run \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new \
  --episode r2r-5351 \
  --credentials /home/xinglinlu/.config/llm/credentials.env

# 将上一条命令的 episode 依次改为 r2r-6244、r2r-6508、r2r-7275。

# 4. 页面、逐任务视频与审核表；不调用API。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot_review \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new
# 检查链接、哈希、视频帧对齐。
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot_review \
  --root /data/ticvla/outputs/stage1a/r2r-pilot-new --audit
```

准备入口也可以单独调用 `semantic_loop prepare --vocabulary <配置路径>`；未提供时仍使用仓库默认词表。室内词表和权重绑定在感知 manifest 中，正式调用时验证哈希，并拒绝没有同帧检测支持的输出类别。

## 失败、完成状态与人工审核

单帧超长、非法格式或无依据完成声明仅隔离当前帧，不裁剪原文、不补问、不入图。下一帧使用最近有效上下文，Local 超过 30 秒失效。API、预算、来源或存储异常停止后续任务。完成过的付费请求不自动重发；不删除账本解除去重。

四个任务的原始执行事件为空。空参考线不等于已完成，也不人为补造 arrival/stop 事件满足校验。无证据 DONE 仍记录拒绝；该不足单列为完成判断证据缺口，不掩饰为完整任务监督已通过。

报告展示全部计划帧的实际输入、原文、格式与规则结果、地图入图/OCR、上下文断点，以及示范 GT。视频按源时间回放，在对应关键帧更新结果和年龄；不是按 API 返回时间播放，不能据此认定在线实时。

人工审核可逐字段选择 pass/partial/fail/unscorable，填写原因和修订，下载 JSON。导出始终 `approved=false`，不会自动触发训练。人工修改与模型原文分开，不倒填已经执行的递归链。

## 本次结果位置

本机目录：`/data/ticvla/outputs/stage1a/r2r-pilot-20261029`。最终以其中 `REPORT.md`、`report.json` 和 `review-audit.json` 为准。

12 帧预检发现可用柜子、门、床、画等候选，也发现台球桌被识别为 bed、吊扇被识别为 lamp 等错误，全部保留。通过的是 mask 尺寸/来源接口，不是检测准确率。试点标签仍须按 [Agent.md](../Agent.md) 审核；没有实例 GT 时不报 mask IoU，不把理想 odom 结果外推为真实环境建图能力。


## 2026-10-29 试点结果

- 40/40 帧真实返回，原运行格式与规则通过 27 帧；13 帧 Local 超过80字符，其中一帧还含当前语法不支持的 `~`。
- 原价估算合计 ¥0.756008，包含拒绝回答；API请求耗时合计约20.52分钟，未计感知准备和审核。
- 完成标记检查发现旧逻辑漏掉正文中的 `DONE`。本轮调用全部结束后已修复，并用原响应独立复核：25/40通过规则，另有8帧需要审核受影响历史上下文。没有重发API、修改原回答或重算原运行地图。
- 4段视频共248个源帧，全部40个关键帧可逐帧查看。人工批准数仍为0；不自动扩批。
- 88项相关回归测试通过；页面链接、源文件哈希与四段视频关键帧对齐审计通过。
- 本机审核服务：<http://localhost:8790/>。若浏览器在另一台机器上，需转发服务器8790端口。

重新打开服务：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m http.server 8790 \
  --bind 0.0.0.0 --directory /data/ticvla/outputs/stage1a/r2r-pilot-20261029
```

完整数值、失败原文、地图和复核记录以运行目录中的报告为准。特别是“规则通过”不等于“标注正确”；当前建议先完成人工审核再决定是否扩大标注规模。

## 后续 Local 90 规则（历史实验不重写）

新建试点默认 `qwen_plus_teacher.v2.json`。Local 必须简洁且最多90字符，加入英文文字/数字/普通标点说明，以及可见的完成证据标志和严格DONE约定。原10月29日试点配置、原文、地图与递归历史保留。独立复核位于 [Local90审核](http://localhost:8790/local90-revalidation/)，通过数由25变为30；这是规则复核，不是新prompt实测或人工批准。
