# R2R v6：剩余 108 个任务标注

用户确认本轮处理已准备训练池中剩余的 **108 个任务、1,154 个关键帧**。不扩展至完整 10,819 个 R2R 训练任务，不重标此前 v4 的 12 个任务，不处理 RxR，不训练或自动批准标签。

产物目录：`/data/ticvla/outputs/stage1a/r2r-v6-remaining108-20261104`。

**恢复执行（2026-11-05）：**用户明确要求继续完成，已在新目录 `/data/ticvla/outputs/stage1a/r2r-v6-remaining108-recovery-20261105` 恢复同一108任务、1154帧。新进度页为 `http://localhost:8793/progress.html`。复用全部感知缓存，prompt及模型设置不变，显式直连；仅对原3个未知请求授权各一次新尝试，保留旧预算预留，其他1151帧首次调用。请求指纹与恢复授权见新目录 `recovery.json`，原目录和旧失败响应状态不覆盖。标签仍需人工审核。

**历史状态（2026-11-04）：**感知准备108/108完成；前三次API请求均180秒超时，0次返回，1151帧未请求，原批次停止。没有自动重发或清空预算预留；无用量回执不能认定实际费用为零。

## 网络故障与修复

实测发现同一进程的 `HTTP_PROXY/HTTPS_PROXY` 指向 `127.0.0.1:7890`，而 `http_proxy/https_proxy` 指向 `127.0.0.1:7892`；Python urllib 选择后者。两个端口都能建立本地TCP连接，但对同一认证模型列表请求，7890返回200、7892超时；直连也返回200。浏览器能访问网站不能证明标注进程的代理路径正常。

标注网络层现已使用 [teacher_transport.py](../ticvla/warehouse/dataset_tools/teacher_transport.py)：北京 DashScope 精确域名 `dashscope.aliyuncs.com` 默认直连；其他服务沿用环境代理。使用独立 opener，不修改全局代理，不自动重试或切换线路。实际策略进入新请求 `input.json.transport` 的审计信息，不进入模型prompt。显式配置 `http_proxy_mode` 可选择 `direct` 或 `environment`，但修改冻结配置需生成新版本与哈希，不改写旧实验。

修复验证：同一调用层的认证GET模型列表返回200，未鉴权且无模型的POST返回400，均在约0.1秒内到达服务。该验证没有模型生成请求，不代表已验证完整视觉生成。网络单测与现有语义/预算回归共38项通过。原批次仍停止，3次未知请求未重发。

诊断材料：产物目录的 `proxy-port-diagnostic.json`、`network-fix-verification.json`。后续恢复需独立记录新执行代码及网络策略，保留旧失败记录与未知费用预留；不能直接重跑已含付费记录的 `run_batch.py`。

2026-11-04 23:01（北京时间）已补充一次独立真实图像生成验证：同一 `qwen3.7-plus`、thinking预算2048及输出上限8192，通过修复后的调用层直连。模型列表GET返回200，图像生成9.07秒返回 `{"status":"ok","visible_object":"sculpture"}`，finish_reason为stop。输入303、输出527 tokens（含501思考tokens），冻结单价估算¥0.004822，已记入共享预算。输出目录 `/data/ticvla/outputs/stage1a/qwen-network-verification-20261104-230142`。该样本仅验证连接与图像生成，不作为导航标注；原3次超时请求未重发，108任务批次尚未恢复。

## 冻结配置与执行

- 教师：`qwen3.7-plus`，北京端点，thinking 开启、预算 2048，输出上限 8192，超时 180 秒。
- Prompt：用户提供的 v6 原文，见 [冻结配置](../configs/dataset_preparation/qwen_plus_teacher.v6.json)。Local 最多 90 字符，Route 最多 600 字符。没有自动修复或二次补问。
- 检测：室内 26 类词表、YOLOE-v8s PyTorch FP16、MobileCLIP；文字为 PP-OCRv5 mobile。
- 每个任务独立递归地图、Route、Local；最多三个任务并行，任务内逐帧串行。无效输出保留并隔离该帧，继续使用最近有效上下文；接口、来源、存储或预算错误停止。
- 教师只输出 Landmark、Local、Route。waypoint/ref_line 来自已核验示范轨迹，仅用于监督和审核，不输入教师。所有文本默认待审核，地图对象保持 tentative。
- `pilot.json` 固定全部任务与关键帧，记录来源哈希；`execution-code-hashes.json` 另存本轮实际执行代码哈希，不覆盖冻结时的记录。

本轮已启动一条单次执行链：感知准备完成后调用 API，随后生成全部任务审核页、视频并检查链接与来源。运行脚本为产物目录内的 `run_batch.py`，它不自动重试。**不要重复启动该脚本，也不要重新执行付费 `run` 命令。**

## 查看进度与审核

```bash
python -m http.server 8792 --bind 0.0.0.0 \
  --directory /data/ticvla/outputs/stage1a/r2r-v6-remaining108-20261104
```

若 8792 已在运行，无需重复启动。浏览器打开 `http://localhost:8792/progress.html` 查看实时计数；最终打开 `http://localhost:8792/`。

- `progress.json`：准备数量、请求/返回数量、规则有效/失败数量、费用估算、当前阶段。
- `annotation.log`：API 标注与任务完成记录。
- `episodes/<uid>/plus/`：实际输入、完整原文、解析、地图及因果审计。
- `episodes/<uid>/review/`：最终全部关键帧审核、人工审核模板及连续任务视频。
- `report.json` / `REPORT.md`：最终覆盖、token、费用、请求往返耗时、地图更新与失败统计。
- `review-audit.json`：最终审核页链接、视频关键帧覆盖和来源哈希检查。

初始估算约 ¥22.43，使用原有共享预算账本，不清空历史费用。最终成本按实际 API 用量和冻结单价计算，包含失败响应与思考 token，不等同供应商账单。`completed_unreviewed` 仅表示请求与报告完成，不能当作语义准确率通过或标签获人工批准。

此前 12 个任务使用 v4，本轮使用 v6。两批可构成训练池覆盖清单，但不能称为同一 prompt 的 120 任务对照实验；人工批准前不启用文本训练监督。
