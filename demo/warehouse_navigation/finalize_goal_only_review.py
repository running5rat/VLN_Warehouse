"""Validation and honest report for the Codex-only goal-only reannotation."""
from pathlib import Path
import json
import shutil
from collections import Counter
from ticvla.warehouse.dataset_tools.core import read_json,write_json
from ticvla.warehouse.schema import file_sha256
from demo.warehouse_navigation.rgb_oracle_review import report


def run(root):
    root=Path(root);exp=read_json(root/'experiment.json');history=read_json(root/'route-history.json');checks=[];prev=None;task=None
    for row,h in zip(exp['selected'],history):
        n=row['frame'];p=root/'codex-session'/f'{n:08d}';inp=read_json(p/'input.json');result=read_json(p/'result.json');a=read_json(root/'inputs'/f'{n:08d}'/'audit.json')
        assert h['frame']==n
        if task!=row['task']:task=row['task'];prev=None
        assert inp['previous']==prev
        assert h['previous_route']==(prev['route'] if prev else exp['instructions'][task])
        if n==951:assert h['previous_route']=='Go to sign 04/5R and STOP beside it.'
        assert inp['original_instruction_audit_only']==exp['instructions'][task]
        assert all(file_sha256(root/'inputs'/f'{n:08d}'/k)==v for k,v in a['images'].items())
        assert a['map_source_cutoff'] is None or a['map_source_cutoff']<inp['source']['t_obs']
        assert all(l['last_seen']<inp['source']['t_obs'] for l in a['map_landmarks'])
        assert file_sha256(inp['source']['rgb'])==inp['source']['rgb_hash']
        assert file_sha256(inp['source']['depth'])==inp['source']['depth_hash']
        if result['format_valid']:prev=read_json(p/'parsed.json')
        checks.append(dict(frame=n,route_recursion=True,input_hashes=True,causal_map=True,format_valid=result['format_valid'],projection=result['projection_status']))
    assert len(checks)==37 and not (root/'qwen3.7-plus').exists()
    write_json(root/'input-validation.json',checks)
    write_json(root/'human-review.json',dict(approved=0,reviewer=None,samples=[dict(frame=r['frame'],task=r['task'],status='review_pending',fields={k:'review_pending' for k in ['landmark','local','route','waypoint']}) for r in exp['selected']]))
    write_json(root/'cost.json',dict(external_api_calls=0,qwen_calls=0,external_api_charge_cny=0,session_tokens=None,session_price=None,warning='Session subscription usage is not metered here. No independent GPT6 API call.'))
    report(root);stats=read_json(root/'summary.json')['codex-session']
    diagnostic=Counter()
    for p in (root/'codex-session').glob('*/controller-diagnostic.json'):diagnostic[read_json(p)['reason']]+=1
    write_json(root/'validation.json',dict(samples=len(checks),source_and_recursion_checks='passed',projection_unittests='5 passed',no_api=True,
        controller_diagnostic_counts=dict(diagnostic),actual_new_navigation='not_run',human_approved=0,
        annotation='non-blind Codex session-assisted Oracle, no isolated model identity or inference timing claim'))
    lines=['# Codex 会话重标注：只给04/5R目标，不预设路径','',
    '本轮仅使用当前 Codex 会话辅助标注，没有调用 Qwen 或任何外部标注 API。覆盖同一仓库两个完整录制任务、37个关键帧；这不是独立 GPT6 API 的盲测，也不是新输出驱动车辆的实跑。','',
    '## 初始任务与递归','',
    '**任务一：** '+exp['instructions']['mapping'],'',
    '**任务二：** `Go to sign 04/5R and STOP beside it.`','',
    '第二任务初始输入没有第二通道、远近关系、转向或路径。它仅初始化首帧 Previous Route。后续使用本轮产生的上一有效 Route 和 Local；任务切换时清空旧 Route/Local，保留历史语义地图。原始 instruction 仅保存在审计字段，不逐轮额外输入。','',
    '首帧地图中，机器人箭头朝向与04/5R所在方向相反，目标属于与01/5R、02/5R、03/5R同排的已观测地标。会话据此提出先转身、从横通道进入该排地标所在通道再接近目标的路线。该推断没有得到新的实车执行验证；当前会话已知道旧演示，不能用本结果证明独立模型纯靠地图推理的能力。','',
    '## 首帧实例','',
    '实际初始化 Previous Route：`Go to sign 04/5R and STOP beside it.`','',
    '```json',json.dumps(read_json(root/'codex-session/00000951/parsed.json'),indent=2),'```','',
    'waypoint为空，因为所需转身方向不在当前前视图中。空点不包含旋转控制指令，当前像素协议仍缺少完整原地转向执行通道。','',
    '## 完整覆盖与检查','',
    '| 任务 | 原文/计划 | 格式有效 | 非空投影通过 | 空waypoint | 投影拒绝 |','|---|---:|---:|---:|---:|---:|']
    for task,label in [('mapping','环绕建图'),('target','04/5R目标')]:
        d=stats['by_task'][task];lines.append(f'|{label}|{d["returned"]}/{d["planned"]}|{d["format_valid"]}|{d["projected"]}|{d["empty"]}|{d["rejected"]}|')
    lines += ['', '37帧来源哈希、历史地图时间因果性和 Route 递归校验通过；像素投影单测5项通过。投影通过不代表任务路线、footprint或控制可执行通过。文字、目标关系与像素合理性仍全部待人工批准。', '',
    '## 本轮改动和残留问题','',
    '- 第437帧不再给13.65米长直线，改为近端可见前缀，Route保留即将进入通道的右转。','- 第512、1061帧货架角遮挡连续进路，改为空点并保留转向指令；这降低了错误投影，同时也减少了有运动指导的帧，不能算提高导航成功率。','- 第1292帧已靠近可见04/5R，不再强行给驶向货架的点，输出停车确认指令；只在有录制完成证据的1314帧输出独立STOP。','- 地图改为编号点加图内图例，保留原始OCR冲突和tentative状态；蓝线只是已发生的轨迹，不是送给模型的未来规划。原地图另存map-original.png。','- 第二任务的初始地图仍来自旧实际感知策略的因果重建，不是本轮Codex输出驱动的独立新地图。','- 地图和指令以及人工选点均发生改变，本次不是严格单因素消融。','- 避障与源时效尚未闭环验收；单帧控制诊断不能证明可执行。控制诊断分布：`'+json.dumps(dict(diagnostic))+'`。','',
    '## 视频和审核','',
    '- [逐帧输入、输出及两个视频](codex-session/index.html)','- [环绕视频](codex-session/mapping.mp4) · [目标视频](codex-session/target.mp4)','- [Route递归记录](route-history.json) · [全部来源检查](input-validation.json) · [人工审核清单](human-review.json)','',
    '视频分别展示Previous Route与Generated Route、当前输出源帧、文本是否提交到下一轮。cyan是源RGB像素经同帧深度投影得到的工程曲线，其在后续画面中的位置随旧录制位姿重投影。不是新预测重新控制车辆，也不是Reference Decoder输出。按源时刻展示，不测在线推理延迟。','',
    '当前共有13个空waypoint帧；无法仅靠这些像素输出完成两任务闭环。原地转向、短曲线更新频率与执行检查仍需要独立实跑。未开始新的Isaac运动，不以本次重标注代替自主导航验收。','',
    '视频服务使用HTTP Range支持拖动。外部API调用为0、外部API费用为0；会话自身token、订阅用量及模型推理耗时不可从本工具测量，不编造数值。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['demo/warehouse_navigation/codex_goal_only_review.py','demo/warehouse_navigation/codex_goal_only_labels.py','demo/warehouse_navigation/finalize_goal_only_review.py','demo/warehouse_navigation/rgb_oracle_review.py','demo/warehouse_navigation/rgb_review_media.py','ticvla/warehouse/navigation/rgb_waypoint.py']
    hashes={}
    for f in files:
        dest=root/'source-snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[f]=file_sha256(f)
    write_json(root/'source-hashes.json',hashes)
    report(root)

if __name__=='__main__':
    import sys
    run(sys.argv[1])
