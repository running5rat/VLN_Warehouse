# 单帧导航语义问答与三维地图

默认使用 `--llm-mode objects-only`：真实 InternVL 对当前 RGB 做一次自由物体问答，程序解析列表后直接送入 YOLO-E，再按同帧深度、标定和源位姿生成三维语义实例。不执行第二次完整结构化推理，不训练，通用物体链路不依赖 OCR、人工 ROI、实例真值或货架目录。推荐显式使用已通过物体问答验证的 `--base-model-only`。实时入口可额外开启 `--shelf-signs`，使用独立的 RGB OCR + YOLO ROI 链路读取安装的货架编号牌。

## 运行

在工程根目录，使用独立推理环境；不要使用 Isaac Sim Python：

```bash
PYTHONPATH="$PWD" /data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.run_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --llm-mode objects-only --base-model-only \
  --output "/data/ticvla/outputs/semantic-frame-$(date +%Y%m%d-%H%M%S)"
```

默认任务：沿仓库通道前进，利用可见设施辅助定位，并识别需要避让的物体。可用 `--task` 传入任务文本。默认模型路径为 `/data/ticvla/models/InternVL3-1B`、`/data/ticvla/dataset/TIC-VLA-model.ckpt`、`/data/ticvla/models/yoloe-v8l-seg.pt`，可通过 `--model`、`--vlm-checkpoint`、`--segmenter-weights` 修改。MobileCLIP 位于 YOLO 权重同目录。

`objects-only` 的实际问答是固定物体问题，`--task` 只作为任务追溯信息，尚未实现动态任务语义筛选。默认 `--object-policy navigation` 为问题附加背景排除要求，并在解析后过滤 floor/ground/wall/ceiling 等通用表面和整体场景类别；地面标线、墙上标牌、门、灯等具体线索保留。所有被过滤类别及原因会保存，深度几何不受影响。`--object-policy all` 恢复原始广泛物体提取，便于旧实验复现。该规则不是对所有类别导航价值的模型评测，也不自动清理旧地图。

程序创建新输出目录和独立地图，只导入指定帧。`objects-only` 的 LLM 输入只有源 RGB 和固定物体问题；几何 MapView 继续生成供显示，不进入问答。自由问答上限160 tokens，截断明确失败。先释放 LLM 再加载 YOLO-E。`--llm-mode structured` 恢复完整接口，其生成上限由 `--max-new-tokens` 控制，默认768。

## 协议与接口

`objects-only` 接受逐行、项目符号和编号物体列表；规范化空白和大小写，按原顺序去重，保留全部合法类别，不限制为三类。显式 `[]` 为空列表，空白、截断或含无法解析行的回答记录失败；不补类，不自动重试或调用第二次 LLM。类别事件由程序绑定源图像和请求 ID，role/reason 留空。`object-answer.json` 保存原始回答，`llm/response.json` 保存解析结果、`llm_calls=1`、`structured_inference_executed=false`。本模式不生成 Route、waypoint、参考线、z_sem 或 SemanticSnapshot，也不保存接口头文件。

以下 v0.6 协议仅用于显式 `--llm-mode structured`：

v0.6 外层保持 `<landmark>...</landmark><local>...</local><Route>...</Route><waypoint>...</waypoint><REFLINE>`。新生成的 landmark 每项包含 `image_id/category/navigation_role/reason`；最多三个类别，role 为 localization/avoidance/both，reason 不超过 120 字符。类别由模型生成，没有固定候选枚举。旧两字段记录仍可解析，既有单实例 ROI/OCR 接口保留。

同一图像的类别按空白与大小写规范化去重，类别直接作为 YOLO 文本提示；角色和理由不作为分割提示或静态性证据。`YOLOESegmenter.segment_instances()` 一次返回所有实例，固定 imgsz=640、conf=0.25、retina_masks=True、max_det=100，达到上限记录可能截断。

`consume_categories()` 将类别请求和实例结果保存在 `segmentation/detections.json`，每个 mask 的 SHA256 一并保存。保存完成后再调用地图消费者；同一输出目录重放时不运行 YOLO，来源/请求变化拒绝重放，实例子事件 ID 保持不变。类别无检测或推理错误有独立状态。地图消费者逐实例校验，拒绝形状、深度或关联错误。

投影复用通用深度过滤，不使用牌子专用四边形和平面假设。保存 mask 像素数、有效内部深度点数、过滤后点数、保留率及体素点数。坐标来自采集时刻；地图持久数组位于源 odom 坐标，可视化变换到源机器人坐标（X 前、Y 左、Z 上）。位置是可见表面中心，不是物体完整中心。

## 输出与验收

- `overview.png`：RGB、彩色实例 mask、语义 BEV、真实模型回答。
- `segmentation.png`、`semantic_bev.png`、`semantic_3d.png`：详细二维与三维可视化。所有实例沿用同一配色；拒绝实例用红色虚线框。
- `observation.json`、`request.json`、`prompt.txt`、`input-images.json`、`llm/response.json`：完整输入、真实响应、模型与来源哈希。`interface-heads.pt` 和快照仅由 structured 模式保存，明确未训练。
- `segmentation/`：不可变推理快照、mask、点云和逐实例处理结果；`map/`：SQLite 与几何数组。
- `report.json`：真实模型链路、类别/实例结果、耗时、重复消费检查和几何占用不变检查。

至少一个真实模型提出的类别产生可投影实例，才能通过端到端验证。空类别或全部拒绝不算成功。失败原文和产物保留，重跑使用新目录。

单帧地标保持 tentative，可视化是当前观测层，不升级为静态定位锚点，也不改变已有 MapView 仅显示 verified 地标的规则。潜在动态类别在记录中标注。选定帧没有人或手推车，不能外推这些类别的效果；没有实例真值，不输出虚构 mask IoU 或绝对三维误差。本入口不验收实时性能、跨帧静态性、导航和参考线质量。

```bash
PYTHONPATH="$PWD" /data/ticvla/envs/warehouse-io/bin/python \
  -m unittest discover -s tests/warehouse -v
```

## 2026-10-09 实测与权重限制

代码和 54 项测试通过，但**指定 TIC-VLA checkpoint 的端到端正例未通过**。三次尝试完整保存在 `/data/ticvla/outputs/semantic-frame-20261009-01`、`-02`、`-03`：第一次理由超长被解析拒绝；后两次格式合法，但类别为动作片段，YOLO-E 未检出实例。已用显式正则限制类别为最多三个英文词、单词最多12字符，理由为最多120个ASCII字符，避免固定版本 JSON 约束器允许 token 跨越 maxLength；没有枚举固定物体答案。

对同帧进行独立直接问答：基础 InternVL 返回货架、箱子、灭火器等物体列表，TIC-VLA 初始化后返回 `<answer>` waypoint。结果在 `/data/ticvla/outputs/semantic-frame-caption-probe.json`。因此增加两个**显式可选**参数，默认仍保留原 TIC-VLA 权重：

- `--base-model-only`：基础 InternVL 权重对照，不加载 TIC-VLA 覆盖权重；报告明确标注 `internvl_base_control`，不算原指定权重通过。
- `--object-evidence`：同一个模型先做一次自由物体问答，再将真实回答连同原图、几何 MapView、任务输入结构化推理。自由问答原文保存在 `object-evidence.json`，不是人工类别列表，也不是真值；最终 YOLO 提示仍只取模型结构化回答的 category。

成功对照命令：

```bash
PYTHONPATH="$PWD" /data/ticvla/envs/warehouse-io/bin/python \
  -m ticvla.warehouse.run_semantic_frame \
  --episode /data/ticvla/outputs/warehouse-loop/live-acceptance-20260929/capture/raw/episode.json \
  --frame 200 --llm-mode structured --base-model-only --object-evidence \
  --output "/data/ticvla/outputs/semantic-frame-control-$(date +%Y%m%d-%H%M%S)"
```

完整成功对照位于 `/data/ticvla/outputs/semantic-frame-20261009-base-evidence`。真实模型提出 `shelves`、`boxes`、`fire extinguisher`；固定 conf=0.25 下仅检出灭火器，置信度约0.681，生成76个三维体素点。前两类明确记录为 no_detection，不隐藏漏检、不降低阈值以制造结果。

灭火器 mask 与可见目标对应；通过源图 mask 回投影与源深度一致性检查，并在重新打开地图后验证重复消费不增加证据。这是来源自一致性，不是独立世界真值精度。完整物体问答约0.69秒、结构化推理约34.62秒、分割与投影约0.84秒；不构成实时性能承诺。

成功图像显式标注基础权重，`specified_weights_acceptance_pass=false`；默认权重未自动切换。直接基础模型结构化问答的另外两次对照仅输出整体场景 warehouse，其产物也保留，但不能作为物体地标识别正例。

## Isaac Sim 运动时使用同一链路

在工程根目录执行：

```bash
bash DynaNav/warehouse_loop/run_live.sh \
  --semantic-backend llm --llm-mode objects-only --base-model-only \
  --output "/data/ticvla/outputs/warehouse-loop/live-llm-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

脚本使用 `/data/ticvla/envs/warehouse-io/bin/python` 启动建图和模型进程，通过 `run_host.sh` 使用宿主 `/data/isaacsim/python.sh` 启动车辆。无需另外打开 Isaac 或激活 Conda。默认打开 GUI，右侧面板显示持续积累的几何和待确认物体；浏览器 `http://127.0.0.1:8765` 同时显示地图与最近处理源帧的 mask。`--headless` 可关闭 GUI，浏览器仍可看。输出目录必须不存在。

`--semantic-backend` 默认仍为 `ocr`；只有显式选 `llm` 才启用新链路。`--base-model-only` 使用已成功的基础 InternVL 对照权重，去掉它会加载原 TIC-VLA checkpoint，其物体问答未验收通过。LLM 默认模式为 `objects-only`。需要完整接口时显式传 `--llm-mode structured`；仅在该模式下，`--object-evidence` 才能开启两次问答。`objects-only` 与 `--object-evidence` 同时指定会立即报参数错误，不启动仿真或模型。

`objects-only` 每次只对一张源 RGB 问答；structured 模式才输入同帧深度生成的几何 MapView。**持续积累的地图用于保存和显示，尚未作为 LLM 的全局地图上下文输入。** 不输入历史 RGB、OCR 地标或实例真值。`request.json` 保存 `map_used_by_llm`、模式、输入快照版本和请求派发时的持续地图版本，避免将几何来源元数据误认为模型实际输入。

实时框架：

1. Isaac 物理目标60 Hz、渲染目标30 Hz、同步 RGB-D 采集目标5 Hz，车辆沿原定路线运动，LLM 不改写车辆控制。
2. `live_mapping.py` 消费每个已落盘帧，更新几何、发布地图；该进程是持续地图的唯一写入者。
3. `live_llm.py::llm_worker` 在独立进程中常驻 InternVL 与 YOLO，按序推理，不每帧重新加载。两者共同占用显存，但不并发前向。
4. `--semantic-hz 1` 是源时间采样上限。最多一个正在处理的请求加一个最新待处理帧；较旧的未发送帧被替换，记录为 `semantic_skipped`，不会漏掉它们的几何更新。1 Hz 采样不等于1 Hz语义完成率。
5. Worker 在独立输入快照中保存响应、mask、投影预览；主进程通过 `apply_llm_predictions()` 重放不可变 mask，按原 RGB 的深度和位姿重新关联到持续地图。预览结果不能绕过持续地图的来源、深度和关联检查。
6. LLM 物体保持 `tentative`，橙色显示，不升级成静态配准锚点。objects-only 不生成 Route、waypoint、参考线或 SemanticSnapshot。structured 模式的这些输出仅作接口追溯，快照仍使用离线接口时钟，不发给控制器。

实时输出位于 `mapping/map/`（持续地图）、`mapping/map.png`、`mapping/semantic.png`、`mapping/status.json`、`mapping/metrics.jsonl`。`mapping/requests/<id>/` 保存每个源帧的输入地图、问题、实际问答原文、推理耗时、mask/三维点、预览和最终 `live-commit.json`。`mapping/model-provenance.json` 保存启动模型配置与哈希。总报告分别记录物理/渲染/采集/几何/语义的运动期间实际吞吐、被替换语义帧数、源帧延迟、有效响应与接受实例数。

LLM 模式不使用货架 OCR 的目录匹配验收。链路通过要求全部几何帧入图，且有真实模型有效输出和被接受的实例，并通过来源回投影自一致性检查；它不等于模型准确率或绝对地图精度通过。语义吞吐低于1 Hz时仍如实报告未达标，不把采样频率当作推理频率。

### 2026-10-10 无窗口绕环实测

输出：`/data/ticvla/outputs/warehouse-loop/live-llm-20261010-01`。使用上述命令的 `--headless --port 8766`，未启用 `--keep-viewer`。车辆绕环完成51.59米，运动89.42秒，采集与几何更新均448帧。运动期间物理59.86 Hz、渲染29.93 Hz、采集5.00 Hz、几何与地图发布4.99 Hz。几何采集到入图延迟P95约0.118秒。GUI面板本次未测。

语义共完成3帧：第0、200、447帧，分别接受3、1、2次实例观测，均通过源帧回投影自一致性检查。自由物体问答分别1.02、0.96、0.58秒；结构化推理分别37.98、58.80、26.48秒。运动窗口内只完成第一帧，实际完成吞吐0.0112 Hz；另外两帧在结束后的排空阶段完成。队列最多2项，88个未发送语义候选被较新帧替换，几何帧未丢弃。`pipeline_pass=true`，`all_requested_rates_met=false`，不能声称1 Hz语义推理通过。所有地标仍为待确认；56项单测通过，日志保存为 `unit-tests.log`。


## 2026-10-10 单次问答验证

固定第200帧输出：`/data/ticvla/outputs/semantic-frame-20261010-objects-only-01`。基础 InternVL 一次自由问答耗时0.690秒，输出8类：shelves、boxes、fire extinguisher、safety lines、wall、ceiling lights、electrical panels、safety markings。直接用这8类提示 YOLO-E，分割0.802秒、投影与持久化0.095秒，得到灭火器和安全标线两份有效观测。未检出类别均记录 no_detection。安全标线 mask 覆盖范围较宽，不能据链路通过宣称分割准确。

报告确认 `llm_calls=1`、`structured_inference_executed=false`、`end_to_end_pass=true`；重复消费不增加证据，几何占用保持不变，新地标都是 tentative。未生成 `interface-heads.pt` 或 `llm/snapshot.npz`。这些耗时来自离线单帧，不代表与 Isaac 同时运行的吞吐。

实时绕环输出：`/data/ticvla/outputs/warehouse-loop/objects-only-20261010-01`，使用 `--headless --port 8766`。绕环51.59米，运动95.25秒，448帧全部进入几何地图。运动期间物理56.20 Hz、渲染28.10 Hz、采集4.69 Hz、几何与地图发布4.67 Hz、语义完成0.378 Hz；本次没有达到5 Hz几何与1 Hz语义目标，GUI面板未测。

共39个语义请求、39次自由问答、35个可解析回答，接受88次实例观测，最终61个待确认地标。36个请求在运动期间完成，其余结束后完成；52个旧语义候选被替换，几何帧未丢失。失败的4个回答包含2次截断和2次带引号/牌号的非法类别行，原文保留，没有补类或触发第二次推理。平均自由问答1.006秒、分割0.103秒、预览投影与持久化0.228秒；还存在输入快照准备和主地图关联/发布开销，不能只用问答时间推断端到端频率。

`objects-only-verification.json` 核对全部39份响应：每请求一次模型调用、第二次推理未执行、仅一张源RGB输入、没有导航数据包或快照文件。`pipeline_pass=true`、`all_requested_rates_met=false`。62项测试通过，包括单帧入口和实时worker实际分支的调用检查，以及源位姿和重复消费测试。

## 实时检测检查与货架编号

```bash
bash DynaNav/warehouse_loop/run_live.sh \
  --semantic-backend llm --llm-mode objects-only --base-model-only \
  --object-policy navigation --shelf-signs \
  --output "/data/ticvla/outputs/warehouse-loop/navigation-$(date +%Y%m%d-%H%M%S)" \
  --keep-viewer
```

Isaac 面板增加“全部检出”和“最终入图”两个图像，显示源帧 ID、类别/编号、置信度、框、mask 和状态。浏览器 `http://127.0.0.1:8765` 同时显示持续地图、全部检测、最终接受的实例；表格列出 LLM 类别到 YOLO 检出/无检测/过滤的结果，以及每个实例的来源、框、置信度、地图地标 ID、拒绝原因。仅检测图显示拒绝实例，最终入图图仅显示 accepted。图像来自最新处理完的源帧，界面显示源时间滞后，不能理解为当前车辆位置的即时分割。

`mapping/inspection.json` 是这组可视化的发布标记；`mapping/detections.png` 是全部检测，`mapping/semantic.png` 是最终接受实例。逐请求目录保存相同图像、`inspection.json`、原始回答和分割结果。文本提示模式的 YOLO 类别是在 LLM 提示集合内产生的，并非两个独立分类器的一致性投票。accepted 表示通过地图写入检查，仍可能是 tentative，不代表已验证定位锚点。

`--shelf-signs` 开启独立编号牌支路：源 RGB 红白牌候选 → 透视校正 OCR → 原图 ROI 提示真实 YOLO-E → mask 与牌子轮廓/深度平面检查 → 主地图关联。OCR 识别 `01/4R` 这样的字符串，YOLO 分割牌子；编号不作为 YOLO 类别名。地图保存 `category=shelf_marker`、`name=01/4R`、三维表面中心及观测来源。相同编号在不同源帧中出现且几何重合，才可按已安装固定货架牌策略升级 verified；编号冲突和歧义关联仍由地图消费者拒绝。

这条支路不要求 LLM 当帧选中标牌，在界面中明确标为 `RGB OCR -> YOLO ROI`，不冒充 LLM 输出。它不额外调用 LLM，即使物体问答失败也可独立处理编号牌。OCR 仅适配本环境已安装的红白牌设计，不是任意仓库文字识别器；不读取货架目录、仿真实例标签或世界真值。默认不开启，可按上述命令显式启用。其 mask 哈希、OCR 原文、来源哈希、最终接受结果保存在逐请求的 `shelf-signs/` 中。

普通物体用于搜索或作为待确认导航线索；人、推车等可移动对象不应升级永久定位锚点。已验证编号牌可供后续配准/定位模块查询，但当前绕环仍由原控制脚本执行，尚未用这些地标闭环导航。

### 编号牌与显示验证结果

最新复验目录：`/data/ticvla/outputs/warehouse-loop/nav-inspection-20261010-03`。448帧全部进入几何地图；49个请求各做一次物体问答，46个回答解析成功；一般物体接受92次观测，编号牌接受15次观测，最终8个不同编号，6个 verified、2个 tentative。verified 来源均是编号牌的重复观测，普通 LLM 类别仍为 tentative。未有被当前背景策略排除的类别进入本轮地标层；`floor markings`、`yellow lines on the floor` 等具体地面线索仍允许保留。`black floor with yellow lines` 指向背景地板，已明确排除。

65项测试通过，包含检测与最终接受视图分离、背景过滤不改变几何、OCR mask 哈希检查、来源不匹配拒绝和重放幂等。`validation.json`、`unit-tests.log` 保存验证依据。末轮无窗口运动期间几何4.73 Hz、语义0.50 Hz，未达5 Hz/1 Hz；此前 GUI 实跑为 `nav-inspection-20261010-02`。`nav-inspection-20261010-01/isaac-panel.png` 保存实际 Isaac 面板截图，最新目录的 `web-view.png` 保存最终浏览器页面截图。三次调试运行及提示词对照原文均保留，没有改写失败答案。

新策略只保守排除通用背景，不保证剩余每个物体都对当前任务有用，也不消除类别别名和所有跨帧重复实例。若目标是稳定定位，应使用已验证、带唯一编号的地标层；查找物体可以使用候选观测层。编号牌的 verified 是现有工程规则通过，不等于世界真值精度或闭环导航已经验收。
