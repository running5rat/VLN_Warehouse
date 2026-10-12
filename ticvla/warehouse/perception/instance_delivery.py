"""Summarize measured instance-selection artifacts, including failed gates."""
import argparse
import html
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from ..mapping.semantic_frame import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    quality=json.loads((r/'comparison.json').read_text());live={};services={}
    for scene in ['original','combined']:
        base=r/f'live-{scene}';d=json.loads((base/'mapping/realtime-report.json').read_text());d.pop('store',None)
        capture=json.loads((base/'capture/capture-report.json').read_text())
        d.update(loop_complete=capture['loop_complete'],capture_frames=capture['frames'],isaac_headless=True)
        records=[json.loads(x) for x in (base/'mapping/metrics.jsonl').read_text().splitlines()]
        groups={kind:[x['service_s'] for x in records if x['module']=='worker_service' and x['kind']==kind] for kind in ['seg','qa','ocr']}
        services[scene]={kind:dict(count=len(v),p50_s=float(np.percentile(v,50)),p95_s=float(np.percentile(v,95))) for kind,v in groups.items() if v}
        d['dispositions']=dict(Counter(x['module']+':'+str(x.get('status','')) for x in records if x['module'].endswith(('skipped','late','failure','disposition','evicted'))))
        d['source_projection_scope']='Saved source-frame masks/depth/poses; no absolute live 3D error or IoU measured.'
        live[scene]=d
    tests=(r/'tests.log').read_text();test_count=int(re.search(r'Ran (\d+) tests',tests).group(1))
    new=quality['arms']['C']['total']
    acceptance=dict(interface_tests_passed=tests.rstrip().endswith('OK'),test_count=test_count,
        actual_vlm_successes=quality['qa']['valid'],actual_vlm_frames=144,
        sign_3d_coverage_improved=new.get('sign_correct_3d',0)>quality['arms']['A']['total'].get('sign_correct_3d',0),
        sign_mask_recall_unchanged=all(v['total']['sign_mask_matched']==80 for v in quality['arms'].values()),
        quality_gate_passed=new.get('wrong_binding_admitted',0)==0 and new.get('false_merges',0)==0,
        usefulness_filtering_validated=False,live_timing_passed=all(d['timing_pass'] for d in live.values()),
        loops_completed=all(d['loop_complete'] and d['geometry_all_consumed'] for d in live.values()),
        map_recovery_passed=json.loads((r/'recovery-report.json').read_text())['passed'],
        overall_passed=False,live=live,model_worker_times=services,
        caveats=['Previously inspected regression set, not blind generalization evaluation.','A/B navigation category prompt vs C mapping-only two-image prompt: whole-scheme comparison.','Container carrier absent from detector vocabulary; class-agnostic surface positives are not correct container classifications.','Unannotated facilities cannot be scored as unrelated predictions.','VLM selected all candidates in 115/117 nonempty frames; independent useful-instance filtering is not established.','Two known wrong text-carrier bindings remain; no static anchors granted.','Initial empty-candidate decoder attempt is retained in sibling attempt-01 directory.'])
    write_json(r/'acceptance.json',acceptance)
    cases=[('color-00000007-target-02.jpg','K4-72：OCR 已正确读取，但牌子与货架 mask 都覆盖文字，归属检查返回 ambiguous。问题在文字绑定，不是 VLM 漏选。'),
           ('original-00000057-target-02.jpg','72/5R：有正确 sign mask 并已入图，OCR 只读到 /5R。不能把部分读数当成完整编号。'),
           ('layout-00000057-target-01.jpg','K5-71：OCR 输出 Ki-71。原始目标较小，字符细节不足是视觉假设；日志能直接证明的是识别字符错误。'),
           ('surface-00000010-target-01.jpg','K4-73：没有匹配的容器实例 mask，文字落入大货架 mask，被错误绑定给货架。VLM 实例选择并未修复承载物缺失。'),
           ('color-00000013-target-00.jpg','更远处的标识仍可能没有分割候选。没有候选时，VLM 不能产生真实 mask；漏检建议仅作诊断。')]
    text=['<h1>建图实例筛选：结论与失败原因</h1>', '<p><a href="index.html">完整逐帧 A/B/C 图库</a> · <a href="comparison.json">逐场景指标</a> · <a href="acceptance.json">验收与实时统计</a> · <a href="manual-review.json">目视核验记录</a></p>',
          '<p><strong>实现与验证已完成；恢复了 sign 准入召回，但整体质量与实时验收未通过。</strong> VLM 不输入导航任务，只做一次源帧候选实例问答。接口测试数量与结果见 tests.log 和 acceptance.json。</p>',
          '<p>三组 sign mask 检出均为 80/320（25%）。116 个可读 sign 编号观测中，旧精确匹配为 3（2.6%），sign/label 映射为 13（11.2%），新实例筛选为 49（42.2%）。这是正确编号三维观测覆盖率提升，不是分割模型召回提高。</p>',
          '<p>直接印字容器另计：A/B/C 为 0/6/11 个正确文字＋几何观测，共 17 个可读目标。固定词表没有 container，不能声称载体类别识别正确。六组中遮挡组没有可读编号真值，OCR 准确率不可评分。</p>',
          '<p>新 VLM 在有候选的 117 帧中有 115 帧全选，且 245 次实例选择中只有 16 次被称为 identity_candidate。不能认为它已经学会挑选独特地标；不相关对象比例因设施标注不完整而不可定量。第 200 帧的货架区域仍被误分为灭火器并获准入图。</p>',
          '<h2>实时绕环</h2><table><tr><th>场景</th><th>几何 Hz</th><th>候选 Hz</th><th>候选 p95</th><th>实际入图帧 Hz</th><th>文字绑定 p95</th><th>替换分割帧</th></tr>']
    for scene,d in live.items():
        text.append(f"<tr><td>{scene}</td><td>{d['geometry_hz']:.2f}</td><td>{d['candidate_hz']:.2f}</td><td>{d['mask_p95_s']:.3f}s</td><td>{d['admission_hz']:.3f}</td><td>{d['ocr_p95_s']:.3f}s</td><td>{d['drops']['seg']}/448</td></tr>")
    text+=['</table><p>两圈使用 Isaac Sim 5.0 headless 渲染沿固定轨迹完成约 51.59 米，几何 448/448 全部消费；候选完成频率低于 4.75 Hz，发布 p95 超过 500 ms，文字绑定 p95 超过 2 秒，因此实时验收失败。正式入图仅计 accepted_instances&gt;0 的帧，空选择不能计为入图。浏览器绘制延迟未测量。</p>',
           '<p>每圈 OCR 实际完成 91/91，运动期间完成 89/91。原始/综合场景分别只有 31/34 个采样帧具有已入图父实例，均完成绑定处理，但仅各 3 个在运动期间且两秒内发布；其余不适用或迟到帧均在 JSON 中列出，不能用小分母掩盖损失。26/33 次文字属性更新并不代表同等数量的正确编号。</p>',
           '<p>已目视核实运动期间的完整读号正例：原始 72/5R；综合场景 K4-71、K4-72、K5-73、K5-72。所有对象仍为 tentative，算法的跨帧文字一致性确认不等于全局唯一或永久静态。</p>',
           '<h2>仍然失败的图片</h2>']
    for image,reason in cases:
        if (r/'review'/image).exists():text.append('<p>'+reason+'</p><img loading="lazy" src="review/'+image+'">')
    text+=['<p>第 200 帧：两个重叠候选位于同一细窄货架构件，另一个覆盖货架上的标记区域，均被检测为 fire extinguisher 并被 VLM 选择。真实的 04/05 标识没有 sign mask；三个候选获准入图不能作为建图准确性正例。</p><img loading="lazy" src="single-frame-200/semantic.png">',
           '<h2>下一步优先级</h2><ol><li>先修复文字承载物推断：区别牌子与货架的嵌套冲突；缺少真实载体时保留未绑定文本，不直接挂给货架。</li><li>提高小标识候选召回，并检查直接印字载体的类别范围；现有 25% 全目标 sign mask 召回已成为上限。</li><li>对 OCR 的缺半段、单字符错误分别增加诊断与后续视角证据；不自动补号。</li><li>候选实例接口可保留，但当前 1B 筛选近似全选；在新开发数据上加入重复设施、警告词、误检框的反例，再比较 prompt 或 SFT。当前不建议把它当成可靠去冗余器。</li><li>投影准备和图像发布从主调度循环分离，减少排队和替换；仍保持原实时目标，完成后使用新数据重新验收。</li></ol>',
           '<h2>现场结果与地图</h2>']
    for scene in ['original','combined']:
        text.append(f'<p>{scene}: <a href="live-{scene}/index.html">运行界面快照</a> · <a href="live-{scene}/mapping/realtime-report.json">实时报告</a> · <a href="live-maps/maps/{scene}/semantic_bev.png">BEV</a> · <a href="live-maps/maps/{scene}/semantic_3d.png">3D</a></p>')
    for scene,frame in [('original',300),('combined',0),('combined',250)]:
        text.append(f'<img loading="lazy" src="live-{scene}/mapping/frames/live-{scene}_{frame:08d}/segmentation.png">')
    page='<!doctype html><meta charset="utf-8"><title>实例筛选验证报告</title><style>body{max-width:1400px;margin:30px auto;background:#101a24;color:#edf2f7;font:17px sans-serif;line-height:1.6}a{color:#75caff}img{max-width:100%}td,th{padding:8px;border:1px solid #53606d}table{border-collapse:collapse}</style>'+''.join(text)
    (r/'analysis.html').write_text(page)
    index=(r/'index.html').read_text()
    if 'href="analysis.html"' not in index:
        index=index.replace('<h1>', '<p><a href="analysis.html">结论、失败原因与两圈实时报告</a></p><h1>',1)
        (r/'index.html').write_text(index)
    print(json.dumps({k:v for k,v in acceptance.items() if k not in ('live','model_worker_times')},ensure_ascii=False))

if __name__=='__main__':main()
