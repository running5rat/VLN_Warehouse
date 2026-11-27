# R2R v4：12个新任务的完整标注

本轮用户确认只处理12个新任务，不扩至全部120个。使用已准备训练池中12个不同建筑的任务；seed42，先排序并打乱建筑，再从各建筑按UID排序的任务中选一个。选择先于模型结果，失败任务不能换样本。

输出：`/data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030`。共144个预定关键帧；使用既有参考线覆盖计划，不删帧凑配额。既有4个比较任务不在本轮中。RGB-D与理想局部odom来自Habitat动作回放，不启动Isaac、不训练、不自动批准标签。

模型沿用 `qwen_plus_teacher.v4.json`：Qwen Plus，北京端点，思考2048、输出8192、超时180秒，Local必须简洁且最多90字符。Route引用的物体/地点关系可由Local、输入图像或已观测地图支持；合理的Previous Route可原样保留。DONE需要既有完成证据，不从末帧或空参考线推断。

词表沿用室内26类。YOLOE-v8s PyTorch FP16及PP-OCRv5 mobile CPU在隔离环境准备；真实分割/OCR与历史几何均缓存。地图、Route、Local按任务独立递归。最多3个独立任务并行，任务内部严格顺序；共享账本锁定预算，不自动重试。响应格式失败跳过当前帧，API/预算/来源或存储异常停止提交后续工作；已发出的请求保留结果和费用。

## 复现命令

在仓库根目录运行，`--output`必须使用新目录。以下命令展示本次已执行路径，不能原目录重复执行付费请求。

```bash
/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_batch initialize \
  --dataset /data/ticvla/datasets/stage1a-20261021/episodes-v4 \
  --selection /data/ticvla/datasets/stage1a-20261021/selection-v3.json \
  --output /data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030 \
  --config configs/dataset_preparation/qwen_plus_teacher.v4.json \
  --previous-pilot /data/ticvla/outputs/stage1a/r2r-pilot-20261029

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_batch prepare \
  --root /data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_batch run \
  --root /data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030 \
  --credentials /home/xinglinlu/.config/llm/credentials.env --workers 3

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_batch report \
  --root /data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030

/data/ticvla/envs/warehouse-io/bin/python -m ticvla.warehouse.dataset_tools.r2r_pilot_review \
  --root /data/ticvla/outputs/stage1a/r2r-v4-batch12-20261030 --audit
```

## 报告与审核

浏览器入口：<http://localhost:8791/>；远程需要转发8791端口。每个任务提供完整关键帧实际输入、原始JSON、mask/OCR、独立地图、GT参考线/waypoint及连续视频。未来几何只在审核侧展示，未送教师。

逐字段人工审核仍是必需环节；格式通过与语义正确分开。报告分母包含失败、未执行和未返回帧；不给尚未审核的标签标注正确率，不补报mask IoU。

运行后 `report.json`/`REPORT.md` 为统计依据，`failures.json` 保存拒绝原因，`all-answers.json` 索引全部原文。`pilot.json`、配置、模型哈希与源文件哈希冻结；`batch-run.json` 记录实际并行墙钟耗时，API累计耗时不能当作并行总时长。

## 本轮实际结果

12个任务的144帧全部请求并返回，无API错误或重试。规则通过133/144（92.36%），拒绝11帧：10帧Local超过90字符，1帧Landmark列出4类。全部仍待人工审核，规则通过率不等于语义准确率。

|任务|规则通过 / 预定帧|
|---|---:|
|r2r-4914|10 / 12|
|r2r-7272|13 / 15|
|r2r-2758|17 / 20|
|r2r-1714|10 / 10|
|r2r-10042|10 / 10|
|r2r-298|6 / 6|
|r2r-6537|4 / 6|
|r2r-6429|11 / 13|
|r2r-9830|7 / 7|
|r2r-6640|11 / 11|
|r2r-4766|8 / 8|
|r2r-6966|26 / 26|

实际用量：输入340,207 tokens、输出277,573 tokens（其中思考264,767）。按冻结基础单价估算¥2.900998，包含失败响应；不是供应商最终账单。API请求往返累计75.31分钟，最多3任务并行的标注墙钟35.34分钟，单次p50/p95为34.52/35.12秒；不含感知准备、报告生成及人工审核。

地图接受393次实例事件，形成216条tentative地标记录；这不是216个已核验正确物体。全部144帧完成证据未验证，没有字面DONE输出，不能据此认为任务完成判断已通过。waypoint/ref_line由示范轨迹产生，不计教师能力。

抽查发现r2r-9830第43/46帧重新引入早先删除的“Exit the room”步骤；r2r-10042的Route累积已走过步骤的总结；r2r-6537的“Waiting”描述需要核对是否有实际停等依据。r2r-4914也展示了由卧室到浴室入口、再到浴室内部的Local变化。完整图片、原文与判断范围在[抽查分析](http://localhost:8791/analysis.html)，不将助手抽查冒充人工批准。

已经交付12份独立地图、12个整段回放视频和144帧审核表。包括以前4个比较任务，目前累计16个不同R2R任务、184个唯一关键帧；120个训练池任务中本轮处理12个，剩余108个尚未走本链路。此前同任务重跑不重复计数。

推荐先审核本批，尤其是Route进度变化、楼梯、目标附近及拒绝帧，再决定扩批。未开始120任务全量标注，未导出自动批准的训练标签。

汇总可从[完整报告](http://localhost:8791/REPORT.md)、[机器统计](http://localhost:8791/report.json)、[交付审计](http://localhost:8791/review-audit.json)查看。批次目录中的`finalize_report.py`生成附加统计和图片分析，无模型调用；重建完整报告时，在上述report、audit命令后运行该脚本一次，再运行audit检查新增链接。

验证：51项针对性测试通过；输入/响应来源哈希、全部关键帧视频对齐、地图因果与重启恢复检查通过。审核页保持人工批准数为0。
