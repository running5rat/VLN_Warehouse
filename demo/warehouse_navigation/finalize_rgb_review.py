"""Freeze review summary without turning automated checks into approval."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from demo.warehouse_navigation.rgb_oracle_review import report
from ticvla.warehouse.dataset_tools.core import read_json,write_json
from ticvla.warehouse.schema import file_sha256


def run(root):
    root=Path(root);report(root);summary=read_json(root/'summary.json');exp=read_json(root/'experiment.json')
    audit=[];tokens=Counter();errors={};sample_status={}
    for row in exp['selected']:
        p=root/'inputs'/f"{row['frame']:08d}";a=read_json(p/'audit.json');s=read_json(p/'source.json')
        assert a['map_source_cutoff'] is None or a['map_source_cutoff']<s['t_obs']
        assert all(lm['last_seen']<s['t_obs'] for lm in a['map_landmarks'])
        for name,sha in a['images'].items():assert file_sha256(p/name)==sha
        audit.append(dict(frame=row['frame'],causal_map=True,source_hashes=True))
    for provider in summary:
        counts=Counter();states=[]
        for p in sorted((root/provider).glob('*/result.json')):
            d=read_json(p);counts[d.get('error') or d.get('projection_error') or d.get('projection_status')]+=1
            states.append(dict(frame=d['frame'],task=d['task'],verdict='review_pending',
                               projection=d.get('projection_status'),format_valid=d['format_valid']))
            if provider=='qwen3.7-plus':
                u=d.get('api_usage') or {};tokens.update({k:u.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
        errors[provider]=dict(counts);sample_status[provider]=states
    write_json(root/'input-validation.json',audit)
    write_json(root/'human-review.json',dict(approved=0,samples=sample_status,
        note='Codex analysis is not user approval. Pixel projection pass is not navigation success.'))
    q=summary['qwen3.7-plus'];c=summary['codex-session']
    lines=['# RGB 像素 waypoint：两任务 Oracle 审核报告','',
        '完成 Qwen 3.7 Plus 真实调用与 Codex 会话视觉代标注。每份报告覆盖同一仓库的两个完整 Instruction：环绕建图，随后前往起点 RGB 不可见的第二通道 04/5R。共37个审核时刻：21个环绕任务时刻、16个目标任务时刻，包含真实记录的两次停止。','',
        '**本次是固定运动回放的输出、深度投影与控制接口诊断，没有用新预测重新驱动车辆。** 原视频的控制来源仍是此前的米制 Codex 脚本。两模型共用严格因果重建的原感知地图，不是各模型自身输出形成的新闭环地图。所有人工批准数为0。','',
        '| 检查 | Qwen 3.7 Plus | Codex 会话代标注 |','|---|---:|---:|',
        f'| 有原文 / 计划 | {q["returned"]}/37 | {c["returned"]}/37 |',
        f'| 格式与文本规则有效 | {q["format_valid"]} | {c["format_valid"]} |',
        f'| 非空像素路线投影通过 | {q["projected"]} | {c["projected"]} |',
        f'| 合法空 waypoint | {q["empty"]} | {c["empty"]} |',
        f'| 格式有效、投影拒绝 | {q["rejected"]} | {c["rejected"]} |',
        f'| 请求 p50 / p95 | {q["request_p50_s"]:.2f} / {q["request_p95_s"]:.2f}秒 | 未测，不是在线服务 |',
        f'| 请求≤5秒 | {q["within_5s_source_age"]} | 不适用 |','',
        '投影统计先要求文本协议有效；文本失败的帧保留原文，但不发布控制。空路线可能表示停车或看不到下一方向，不能算成功导航。投影通过只说明点落到可测地面、拟合曲线仍可见，不能证明 Route 正确、footprint 可通过或任务完成。','',
        '## 输出问题与证据','',
        '- **Qwen 的点经常在图内、却不在地面。** 197帧等给出 y=250/350 的点，落在墙面或无有效深度区域。437、497、1074、1136、1196等也有相似问题。原文、像素叠图、深度点与具体拒绝原因都保留。','',
        '- **顺序不一定是近到远。** 317帧给出 v=600→650→700 的序列，实际向相机附近回收；377、427等点偏向画面下边缘。即使某些通过地面投影，也不能因此视作符合 Route 的正确路径。','',
        '- **Qwen 提前删掉未完成步骤。** 437帧仍在远端横通道，Route 已删掉进入第二通道的右转；497帧进一步简化为返回起点。后续继承这个错误，不能把措辞缩短当作任务进展。','',
        '- **第二任务没有稳定完成地图推理。** 951帧地图已经含 identity_confirmed 的04/5R，投影视锥检查显示它在当前RGB之外。Qwen仍只复述目标，未给出完整跨通道步骤。之后部分帧能够读到目标，但存在提前停止声明。地图标签重叠、图例可读性差是输入侧的实际限制，不能把全部错误归于模型。','',
        '- **Codex也有错误。** 512与1061两次转弯的手选像素落到了高于地面的物体表面，被同一投影规则拒绝，未自动挪点补成成功。需要转身、目标暂不可见时保留空点。Codex在当前会话中看过既有演示和整段回放，属于非盲的人工式Oracle参考，不等同于同输入的Codex API独立测评。1314的Local草稿澄清另存修订记录。','',
        '## 实时性与接口边界','',
        '在线代码新增每2秒调度、一个处理中＋一个最新等待帧，不等待旧曲线走完。新有效曲线可立即接管；源年龄超过5秒拒绝发布，旧指导超过5秒失效。Codex会话回答通过冻结文件请求交换，不伪装成有毫秒级延迟的模型服务。','',
        '本次Qwen没有一次请求在5秒内完成，因此不满足这里的实时控制配置；其中831帧耗时约171秒。未拆出服务端prefill/decode，所有数字都是请求往返时间。没有实际新协议Isaac闭环验收，也没有为通过而延长过期阈值。','',
        '纯像素路线还缺少明确的原地转向表达：下一条通道不在RGB内时，空waypoint只会停车。不能沿用旧脚本隐藏的地图点完成转身。该问题需要单独确定观察/转向接口后再做完整自主环绕。','',
        '## 输入、输出和曲线','',
        '输入图像顺序：当前960×720 RGB、同帧分割叠图、此前的语义地图。无历史RGB图片；离线输入包含上一有效Local和相对运动。初始Instruction只初始化首帧Previous Route；其后使用该模型最近有效Route。坏文本帧跳过、保留原文，不补问。两个任务重置Route和Local，继续使用此前地图。终止补充帧没有同帧检测，类别列表为空，不能复用邻帧mask。','',
        '输出为landmark、local、route、五个RGB像素waypoint或空列表。像素以同帧深度、K、相机外参和源odom投到3D；地图坐标用于显示与控制对齐。源码保留旧米制协议兼容，绝不把旧米制点标成像素。PCHIP是工程控制曲线，Reference Decoder与FMAE本轮没有运行。','',
        '数据来自原Isaac Sim 5.0静态仓库、真实YOLOE-v8s TensorRT/PP-OCRv5 mobile结果及仿真理想odom。共同地图按源时间回放重建，没有把最终地图回填到早期。末帧的完成证据来自原实跑日志；不是本轮Oracle真的控制到达。','',
        '## 费用与审核入口','',
        f'Qwen累计输入 {tokens["prompt_tokens"]:,} tokens，输出 {tokens["completion_tokens"]:,} tokens。按冻结配置输入2元/百万、输出8元/百万计，未折扣估算约 **¥{q["estimated_CNY"]:.4f}**；包含已返回的失败标注，不是账单价格核验。思考预算2048、最大输出8192、北京端点、单请求180秒超时。Codex会话未调用独立API，不报告虚构的API token或价格。','',
        '- [Qwen完整两任务报告](qwen3.7-plus/index.html)','- [Codex完整两任务报告](codex-session/index.html)',
        '- [冻结SYSTEM](system.txt) · [数据清单](experiment.json) · [来源校验](input-validation.json)',
        '- [接口检查与源码哈希](implementation-validation.json) · [人工审核状态](human-review.json)','',
        '每个模型提供一任务一视频、全部逐帧输入与原文、RGB像素点、BEV/3D深度诊断、规则失败和控制器诊断。视频按源帧时间更新，不用来声称在线低延迟。12秒基础抽样加转弯帧属于离线评审选点，不是线上2秒调度；coverage.json另列参考前缀长度不足的区间。','',
        '结论：可以审核新的像素输出及同帧投影接口；Qwen尚不适合直接作为本配置的实时控制Oracle。Codex参考标注也不能代替新协议自主导航闭环验收。','',
        '## 全部失败分类','', '```json',json.dumps(errors,ensure_ascii=False,indent=2),'```']
    (root/'REPORT.md').write_text('\n'.join(lines))
    write_json(root/'cost.json',dict(usage=dict(tokens),estimated_cny=q['estimated_CNY'],rates_cny_per_million=dict(input=2,output=8),
        source='frozen qwen_plus_teacher.v6.json; estimate not current invoice',codex_api_cost=None))
    # All planned samples get review rows; failures are not removed from videos.
    report(root)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
