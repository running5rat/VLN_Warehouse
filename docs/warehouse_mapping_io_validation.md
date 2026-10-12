# v0.6 地图与 LLM I/O 阶段验证

验证日期：2026-09-28。范围按本轮约定：语义地图工程正确性与真实 LLM 协议链路。
不训练，不验收 LLM 语义/规划正确性、导航控制或实时部署。

## 环境与证据

- Isaac Sim：`/data/isaacsim`，版本 `5.0.0-rc.45+release.23960.184afb15.gl`，宿主机 Ubuntu 22.04。
- GPU：RTX 4090；模型处理使用 `/data/ticvla/envs/warehouse-io/bin/python` 独立环境。
- YOLO-E：Ultralytics 8.3.203，`yoloe-v8l-seg.pt`。
- YOLO-E 权重 SHA256：`14e9c146f7f7efa0bfea8c5f82952c9f95fb1f9c3bf0f18b3727b76624f9d746`。
- LLM：本地 InternVL3-1B，导入 TIC-VLA 的匹配 VLM 参数；新参考线及压缩头仅作接口检查。
- 所有验证资产位于 `/data/ticvla/outputs/warehouse-map-io`。综合报告为 `validation.json`，
  包含分阶段报告路径/哈希和实现源码哈希；环境版本清单为 `environment-freeze.txt`。

## 关卡与结果

| 阶段 | 实际验证 | 结果/证据 |
|---|---|---|
| V0 | 既有回归和新增地图/协议测试；Isaac同步采集 | 30项单测通过；13帧，源时间跨度3.10秒，采集进程退出0 |
| V1 | RGB-D投影与独立USD箱体表面比较 | Oracle和真实分割的表面距离P95均≤0.00641 m，低于0.10 m阈值 |
| V2 | 首尾两帧、三个对象的真实YOLO-E分割与建图 | 6/6入图；mask IoU 0.97176–0.99204；表面中心对照误差≤0.00154 m |
| V3 | 保存/重载、跨任务关联、去重、拒绝、odom原点变换及撤销 | 通过；原点回放平移误差0.02155 m，yaw误差0.004155 rad |
| V4 | TEXT和PREFILL_FOLLOW真实权重前向、格式与张量验证 | 通过；TEXT生成288 tokens且解析有效，PREFILL生成0 tokens；两种快照形状正确 |
| V5 | 真实输出→来源解析→语义事件→真实分割/明确拒绝 | 通过派发验证；3个对象事件因无检测被拒绝，1个关系事件明确拒绝，不污染verified地图 |

地图证据：

- `capture-acceptance/capture-report.json`：版本、帧数、来源时间及夹具脚本哈希。
- `mapping-verified/report.json`：逐对象IoU、独立表面误差、重定位误差和拒绝原因。
- `mapping-verified/map/map.sqlite`：部署地图；`oracle-map` 是独立特权对照。
- `mapping-verified/view/map.png`、`overlay-0.png`、`overlay-12.png`：地图与真实mask叠图。
- `registration-map`：破坏性撤销测试的独立副本；主地图未因撤销实验丢失数据。

LLM证据：

- `io-verified/observation.json`：实际输入，含四帧RGB及地图。数据库地标UID不进入模型文本。
- `io-verified/text/response.json`：实际约束解码文本、解析结果、token约束计数和请求元数据。
- `io-verified/prefill_follow/response.json`：无生成文本、无语义事件、Route保持不变。
- 两个模式各有 `snapshot.npz`，包含 `[32,2]` 参考线、`[32]` 前缀mask和 `[16,512]` 上下文。
- `io-verified/interface-heads.pt`：固定种子的未训练新头，用于复现接口输出。

综合报告 `validation.json` 的 V0–V5 均为通过，`passed=true`。最终TEXT的本机
服务时长为93.68秒，PREFILL为0.220秒，仅作本次接口运行记录，未做性能验收。
输入历史帧的实际年龄为3.10、2.10、1.067和0秒，没有伪造精确的整秒采样。

## 结论边界

这里的毫米级几何数字来自无传感噪声的固定合成场景与同源深度，不代表实机精度。
重定位使用真实RGB-D/mask加人工坐标原点变换回放，不是物理搬移、VIO漂移或未见场景测试。
手工核验ROI与静态性只用于地图验收；没有把仿真实例ID或真值坐标送入真实分割/LLM。

LLM结构合法来自真实模型在语法约束下选择token，不能解读为自然生成格式能力达到99%，
也不能解读为地标/Route/waypoint正确。模型仍可能产生无意义描述，检测不到的对象会被
拒绝写入地图。新头没有训练，参考线置信度不是安全概率。两个VLM模式分别使用相同
观测验证接口，没有开展跟随任务。首次TEXT服务时长包含tokenizer语法索引构建，
不是在线推理性能验收。

本轮没有实施模型训练；单测中原有微型随机网络的梯度/优化器检查仅属回归测试。
复现命令及后续差距见 [使用说明](warehouse_mapping_io.md)。
