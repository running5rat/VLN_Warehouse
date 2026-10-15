# 第一阶段实验报告（2026-10-13）

本轮实现自然语言入口、结构化递归 Route、执行事实、参考线一致性检查和几何子目标接口，未训练模型。

**结论：接口贯通不等于自然语言导航能力通过。基础 InternVL3-1B 在当前完整协议下存在任务误解、协议文本复制和空 waypoint；本轮不将这些输出替换为人工计划。**

## 确定性验证

148 个仓库测试通过，包含既有回归及新增的协议兼容、参数/来源错误、过期/乱序、Route 回传、原始 instruction 不变、锁定位姿、路口去重、参考线距离/顺序、无证据完成拒绝、PREFILL 完成状态隔离、恢复、倒退诊断和 reference-only 低层命令隔离。

测试日志：`/tmp/phase1-tests-final.log`。部署模型没有训练；测试集中既有的小模型训练回归只验证原训练代码兼容性。

## 六类真实 VLM 递归试验

最终离线目录：`/data/ticvla/outputs/navigation/phase1-recursive-20261013-b`。

输入：真实采集 `live-acceptance-20260929` 第 200 帧、0/1/2/3 秒可用历史和已观测 MapView；每组一个中文原始 instruction，三轮真实 TEXT。后两轮执行事实明确标为离线注入夹具，图像固定，不能当作真实运动、路口识别或行人跟踪证据。

| 项目 | 结果 |
|---|---:|
| 实际 TEXT 请求 | 18 |
| 格式合法、Route 提交 | 18/18 |
| 后一轮实际复用上一轮 Route | 12/12 |
| 有限 z_sem / Decoder / 九模态 Flow | 18/18 |
| 非空 waypoint | 0/18 |
| TEXT 参考线一致性通过 | 0/18 |
| 导航 TEXT 地标入图正例 | 未通过：landmark 均为空 |

“回传 12/12”只证明程序接线，不代表 12 次任务进度正确。查看逐轮原文：模型基本重复上一轮回答，没有正确反映执行事实中的进度缩减。

| 组别 | 实际输出表现 | 判断 |
|---|---|---|
| 地标 | 剩余文本保留 72/4R，但 current_subgoal 把 category 写为 intersection，将 target_text 混入 description | 可解析；未形成正确可查询目标 |
| 顺序路线 | 输出接近“到路口左转，再沿右侧通道”，丢失最终 72/4R；用 approach_landmark 表达 intersection | 原任务保持和动作解释失败 |
| 路口计数 | 文字提到第二个路口，但动作是 approach_landmark；经过一个路口后不正确缩减 | 递归进度失败 |
| 探索 | 复制协议短句 `approach_landmark: target_text and/or category/description`，动作为 advance_to_intersection | 任务理解失败 |
| 跟随 | 文本写 Follow the pedestrian ahead，动作却是 approach_landmark / person | 未进入正确跟随状态机 |
| 倒退 | 复制协议短句并输出 advance_to_intersection；没有后向 waypoint | 后向任务表达失败 |

倒退三轮夹具期望剩余 5、3、1 m。原 Decoder 终点分别约 `(0.748,-0.681)`、`(0.541,-1.110)`、`(0.199,-1.818)` m，x 均为正，不是后向终点。原始张量保留；没有用插值线替换，没有执行物理倒车。这是当前未训练头的诊断，不是训练后的精度评价。

### 离线耗时

RTX 4090，模型常驻，不含启动加载；固定这 18 次请求。

| 阶段 | p50 | p95 |
|---|---:|---:|
| 图像编码 | 96 ms | 120 ms |
| 整体文本生成（含 prefill） | 1,199 ms | 1,484 ms |
| prefill GPU | 106 ms | 122 ms |
| decode GPU | 1,012 ms | 1,261 ms |
| 原 Decoder＋上下文压缩 | 13 ms | 16 ms |

整体生成与其内部 GPU 分项不能相加。Flow 头耗时、每轮完整服务时间及实际输出形状见逐轮 report.json。本轮未做部署实时验收。

### 调试历史

`phase1-recursive-20261013-a` 保留首次实验。初版 Route grammar 使用通用参数对象，允许输出动作无关字段，严格解析拒绝了这些结果。已改为按动作区分的联合 schema，仍由真实模型选择动作及参数。`b` 的语法通过率提高没有改善任务理解；没有将约束修正描述为模型能力提升，也没有二次修复问答。

首次在线 `phase1-online-explore-original-a` 暴露 Route 日志文件变量覆盖采集流的问题；保留失败日志。修复后的 `...-b` 完整消费 123/123 帧、提交 8 次 TEXT、执行 111 批 Flow，车辆无前进命令，任务以 budget_exhausted 结束。

## Isaac Sim 在线矩阵

结果目录：`/data/ticvla/outputs/navigation/phase1-isaac-matrix-20261013`，含每次完整命令、独立地图、实际 RGB-D、请求响应、Route 历史、候选与实测命令。

原仓库与独立路口场景分别运行地标、路线、探索、跟随的几何对照和模型模式，另有原仓库的静止倒退诊断。每次物理任务预算为 24 秒；预算内未完成不能推广为长时间必然失败。

独立路口场景是无编号牌子的合成拓扑夹具，其地标请求属于“目标未观测”负例，**不是已观测目标到达正例**。已有地图地标正例尝试使用原仓库真实地图，并通过显式重叠图像及几何验证配准；没有给模型目标真值坐标。

17/17 次实验正常退出，2088 帧全部消费，提交 132 次 TEXT，执行 1788 批 Flow。参考线一致性通过 0 次，实际非零轮速命令 0 次，任务均以 budget_exhausted 结束。所有导航 TEXT 的新地标入图正例均未通过。

| 场景 | 任务 | 后端 | 几何帧 | TEXT | Flow 批数 | 实际移动 | 结束状态 |
|---|---|---|---:|---:|---:|---|---|
| warehouse | landmark | geometry-baseline | 123 | 5 | 98 | 否 | budget_exhausted |
| warehouse | landmark | model | 123 | 4 | 86 | 否 | budget_exhausted |
| warehouse | route | geometry-baseline | 123 | 7 | 103 | 否 | budget_exhausted |
| warehouse | route | model | 123 | 7 | 106 | 否 | budget_exhausted |
| warehouse | explore | geometry-baseline | 123 | 7 | 102 | 否 | budget_exhausted |
| warehouse | explore | model | 123 | 8 | 108 | 否 | budget_exhausted |
| warehouse | follow | geometry-baseline | 123 | 8 | 106 | 否 | budget_exhausted |
| warehouse | follow | model | 123 | 7 | 105 | 否 | budget_exhausted |
| junction-fixture | landmark | geometry-baseline | 123 | 9 | 112 | 否 | budget_exhausted |
| junction-fixture | landmark | model | 123 | 9 | 105 | 否 | budget_exhausted |
| junction-fixture | route | geometry-baseline | 123 | 9 | 109 | 否 | budget_exhausted |
| junction-fixture | route | model | 122 | 8 | 106 | 否 | budget_exhausted |
| junction-fixture | explore | geometry-baseline | 123 | 9 | 110 | 否 | budget_exhausted |
| junction-fixture | explore | model | 122 | 9 | 105 | 否 | budget_exhausted |
| junction-fixture | follow | geometry-baseline | 123 | 9 | 112 | 否 | budget_exhausted |
| junction-fixture | follow | model | 122 | 9 | 107 | 否 | budget_exhausted |
| warehouse | reverse | reference-only | 123 | 8 | 108 | 否 | budget_exhausted |

以上 TEXT/Flow 总数包含停采后有界归档；实际运动期间频率分别保存在各 report.json，不能用总数冒充实时频率。

## 未通过与后续缺口

- **Route 语义正确性未通过**：需要区分动作、指代字段、最终目标保持、路口计数和剩余距离。当前证据不能断言该模型在所有 prompt 下都无法理解，只能说明本轮固定模型与完整协议组合失败。
- **参考线质量未通过**：空 waypoint 与非空原 Decoder 线不一致；新控制门禁止将其作为 TEXT 运动指导。Decoder、压缩器、Flow 仍未训练。
- **四类工程任务完成未通过**：不能拿历史人工指定 task-mode 的移动结果替代新自然语言入口的结果。独立路口的已观测地标到达正例仍缺失。
- **导航 TEXT 语义事件入图正例未通过**：真实输出 landmark 为空，未用手工类别补成成功。原建图入口保持兼容。
- **跟随丢失/恢复未通过本轮自然语言端到端验收**：初始动作解释错误时尚未进入绑定，不能把零运动算作“丢失后正确停车”。已有连续性与丢失停止接口单测通过。
- 当前通道/路口提取为局部已观测自由空间的保守几何候选，尚未验证复杂拓扑稳定性；视觉属性指代没有通用检索器，无法唯一匹配时暂停。

下一阶段应先针对这批实际错误建立“instruction＋旧 Route＋执行事实 → 新 Route/动作/waypoint/地标”的监督数据和独立验证集，再分别处理参考线与 Flow 训练。不能只继续增加语法约束，或把几何对照结果写回模型输出。

机器可读汇总已保存在仓库：[navigation-phase1-summary.json](validation/navigation-phase1-summary.json)。

使用说明与全部复现命令见 [第一阶段实现说明](warehouse_navigation_phase1.md)。汇总页面和 JSON 位于 `/data/ticvla/outputs/navigation/phase1-summary-20261013/`。建议在 `/data/ticvla/outputs/navigation` 启动 HTTP 服务，访问其 `phase1-summary-20261013/index.html`。

## 最小补齐验证（2026-10-13）

本次仅增加可读审计文件、TEXT 分阶段墙钟计时、离线小样本选择及工程隔离检查；不改变在线 prompt 默认值、任务拆解方式或模型控制来源，不训练。

- 在线与离线请求新增 `prompt.txt`、`answer.txt`、`audit.json`，有参考线时导出全部原始点到 `reference.csv`，包括 mask 无效后缀。
- TEXT 记录输入／视觉／生成 token 数、首 token 墙钟、后续 decode 墙钟、语法处理墙钟、最终查询前向和头部耗时。首 token＋后续 decode 等于整体生成；GPU 和语法时间是内部项，不能重复相加。
- 新增三个测试，覆盖可读审计保真、正确前进再转向子目标的运动学反馈、watchdog 正常放行与取消／过期／新障碍／epoch 拒绝。全仓库测试 **151/151 通过**。运动学夹具不是 Isaac 轨迹，也不是 VLM 正例。

真实 InternVL 对照固定第 200 帧及相同历史／地图，地标、探索、倒退各两轮，每轮一次 TEXT：

| 项目 | 原 prompt | 精简 prompt（仅离线） |
|---|---:|---:|
| 接口解析通过 | 6/6 | 4/6 |
| 非空 waypoint | 0/6 | 0/6 |
| 非空 Landmark | 0/6 | 0/6 |
| 参考线一致性通过 | 0/6 | 0/6 |
| prefill GPU 中位数 | 105 ms | 89 ms |
| 首 token 墙钟中位数 | 107 ms | 90 ms |
| 后续 decode 墙钟中位数 | 1,088 ms | 1,045 ms |
| 整体生成墙钟中位数 | 1,195 ms | 1,135 ms |

六次统计包含第一次调用效应，不是吞吐或实时验收。精简版地标回答能将 `72/4R` 放入 target_text，并使用 label 类别；但两次探索输出空 target_text，被解析器以 `Invalid target description` 拒绝。倒退仍误输出 approach_landmark。故保留原 prompt 作为部署默认，未用精简版替换失败回答，没有自动修复问答。

另外使用**旧 `--task-mode explore`＋`geometry-baseline`** 在真实 Isaac Sim 5.0 运行 20 秒预算：采集及几何消费 **103/103 帧**，实测移动 **0.520 m**，443 次非零平移轮速命令检查记录，4 次 TEXT 提交、68 批 Flow。5 个实际在线响应均生成审计侧文件，逐项核对 prompt／answer 与 response.json 一致。以 `budget_exhausted` 结束，实时目标未通过。这证明旧工程探索路径与实际命令／观测回流仍可运行，不代表自然语言任务完成或模型驱动运动通过，也未验证连续完成两个前沿。

在线复现命令：

```bash
bash DynaNav/warehouse_loop/run_navigation.sh \
  --task-mode explore --execution-backend geometry-baseline \
  --budget-seconds 20 --max-steps 1560 --headless --port 8774 \
  --output "/data/ticvla/outputs/navigation/phase1-engineering-$(date +%Y%m%d-%H%M%S)"
```

结果目录：

- `/data/ticvla/outputs/navigation/phase1-minimal-20261013-summary/index.html`：结果导航；同目录有 summary.json、unit-tests.log。
- `phase1-minimal-20261013-original`、`phase1-minimal-20261013-compact`：真实离线请求、原文、数值和逐轮报告。
- `phase1-minimal-20261013-isaac-explore`：真实 Isaac 采集、地图、执行命令与报告。

这次最小补齐没有把第一阶段标为全部通过。自然语言任务完成、导航 TEXT 地标入图正例仍未通过；真实地标到达、顺序路口动作和跟随丢失／恢复仍需工程正例，不能全部归因于训练。第二阶段可开始少量监督与 Oracle 建设，但批量采集前应验证对应执行底座。Decoder／压缩器／Flow 的导航质量留给后续监督与训练，不在本次继续扩展控制功能。
