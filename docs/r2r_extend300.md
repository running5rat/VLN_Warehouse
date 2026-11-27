# R2R 标注累计300任务：新增176任务

2026-11-13用户授权继续标注至总计300个不同任务。已有120个训练池任务与4个试点任务，本轮新增176个训练任务、2,012个关键帧；完成后为296个训练池任务加4个比较任务，不将试点自动并入训练。

输出：`/data/ticvla/outputs/stage1a/r2r-extend300-20261113`。

## 冻结配置

- Qwen `qwen3.7-plus`，北京端点，thinking开启、预算2048、输出上限8192、超时180秒；HTTP直连。
- 继续使用 `qwen-plus-v6-route-only-preserve-v1`，SYSTEM正文哈希与旧120任务完全一致。只更新本轮请求授权ID，不换prompt。
- 原始instruction初始化首帧Previous Route；后续递归传入最近有效Route和有效期内Local。保留全部未完成步骤及最终停等要求。
- 教师仅生成Landmark、Local、Route；waypoint/ref_line为示范轨迹监督，不输入教师。
- 既有室内26类YOLOE-v8s和PP-OCRv5 mobile；每个episode独立语义地图。未批准标签不进入正式训练。
- 新清单固定seed42，建筑轮转、按回放长度三个区间均衡，覆盖60栋建筑。排除旧124任务及其路线组、比较任务路线组；本轮不重复路线组，不按模型结果换样本。

## 执行状态和预算

执行器是 `ticvla.warehouse.dataset_tools.r2r_extension_run`。顺序为全部真实感知准备、最多3个任务并发标注、审核页与连续视频、来源/链接检查。任务内严格按时间递归。

按旧120任务实际token用量和冻结价格估算本轮约¥23.88；不是供应商实时账单。原保守预算剩余36.3038705，保持100上限和全部历史预留。每次请求继续经过共享预算保护，不承诺费用必然不超估算；到预算边界停止新请求。

首次启动已记录在 `worker-process.json`，独立后台进程继续工作。**不要重复启动付费批次。** 传输/预算/来源/存储错误停止，失败原文保留，无自动重试和模型替换。再次执行检测到旧运行产物会拒绝，需要先检查中断状态。

- `progress.json/html`：感知准备、请求、返回、新增及累计完整返回任务数。
- `statistics.html`、`task-statistics.json/csv`：逐任务和逐帧规则检查。
- `perception.log`、`annotation.log`、`worker.log`：阶段日志。
- `episodes/<uid>/bundle`、`perception`、`plus`、`review`：完整输入、模型原文、地图、审核和视频。
- `completion.json`：只有完整流程完成后才写入；没有此文件不能据启动日志宣布300任务完成。

准备期间响应数为0是正常状态。全部返回也不等于全部语义正确；人工批准数仍为0。4个试点与本轮prompt版本不同，累计覆盖统计不能当作同prompt300任务消融。

## 查看

当前浏览器入口：`http://localhost:8805/progress.html`，每30秒刷新。HTTP服务支持视频Range请求；重启服务无需重启标注：

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.video_server \
  --directory /data/ticvla/outputs/stage1a/r2r-extend300-20261113 --port 8805
```

已核验176任务的schema、来源哈希与参考线覆盖；批次回归4项通过。只读模型列表接口200且列出指定模型，但该检查不证明生成质量或账户余额。最终验收以实际返回、逐字段规则、递归检查和人工审核分别报告。
