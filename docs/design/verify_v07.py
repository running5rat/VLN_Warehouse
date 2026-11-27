#!/usr/bin/env python3
"""Verify and rasterize the generated document; does not run model code."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '570d0e83bc92880cc3b4cb7d21fcaece6843ecdb924b5ded1a43d1da29265f36'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--renders', type=Path, default=Path('/tmp/vln-v07-qa'))
    args = ap.parse_args()
    args.renders.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'dynamic-warehouse-vln-design-v0.7.md'
    path = ROOT / 'dynamic-warehouse-vln-design-v0.7.pdf'
    pdf = pymupdf.open(path)
    build = json.loads((ROOT / 'docs/design/v0.7-build.json').read_text())
    texts, overflow, links = [], [], []
    for n, page in enumerate(pdf, 1):
        texts.append(page.get_text())
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                        overflow.append({'page': n, 'text': span['text'], 'bbox': span['bbox']})
        links.extend(page.get_links())
        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25)).save(args.renders / f'page-{n}.png')
    toc = pdf.get_toc()
    chapters = [entry for entry in toc if entry[0] == 1 and re.match(r'^\d+\.', entry[1])]
    text = ''.join(texts)
    toc_text = re.sub(r'\s+', '', ''.join(texts[1:chapters[0][2]-1]))
    headings = re.findall(r'^#{2,5} (\d+\..+)$', source.read_text(), re.M)
    architecture = pymupdf.open(ROOT / 'docs/design/figures/model-pipeline-v07.pdf')
    architecture_text = ''.join(p.get_text() for p in architecture)
    # Trace includes clipped text that normal page extraction does not expose.
    architecture_all_text = ''.join(chr(c[0]) for p in architecture
                                   for span in p.get_texttrace() for c in span['chars'])
    displayed_waypoints=re.search(r'<waypoint>(.*?)</waypoint>',architecture_text,re.S)
    pair=json.loads((ROOT/'docs/design/figures/warehouse-assets/global-map-provenance.json').read_text())
    checks = {
        'general_preprocessing_table': '| 处理内容 | 处理方法（含保存内容） | 标准化后的标准格式 | 标注训练用途 |' in source.read_text(),
        'one_annotation_figure': source.read_text().count('<!-- diagram:annotation -->')==1 and '实际 SYSTEM Prompt' not in source.read_text() and '| 项目 | 固定设置 | 作用 |' not in source.read_text(),
        'annotation_images_match': all(sha(ROOT/a['path'])==a['sha256'] for a in json.loads((ROOT/'docs/design/r2r-annotation-example.json').read_text())['figure_images']),
        'removed_original_3_2': '多源任务的统一表示' not in source.read_text() and '3.2.3 因果地图与数据划分' not in source.read_text(),
        'requested_semantic_map': 'demo/semantic_mapping/original-20261013/semantic_bev.png' in build['illustration_assets'] and 'original-20261013' in (ROOT/'docs/design/demo_figures.py').read_text(),
        'actual_r2r_prompt_example': sha(ROOT/'docs/design/r2r-annotation-example.json')==build['annotation_example_sha256'] and all(re.sub(r'\s+', '', item) in re.sub(r'\s+', '', ''.join(p.get_text() for p in pymupdf.open(ROOT/'docs/design/figures/annotation-demo-v07.pdf'))) for item in (json.loads((ROOT/'docs/design/r2r-annotation-example.json').read_text())[k] for k in ('instructions','user_text'))),
        'deployment_agx_orin_64gb': 'NVIDIA Jetson AGX Orin 64GB' in text and 'Orin NX' not in text and '目标部署：NVIDIA Jetson AGX Orin 64GB' in (ROOT/'docs/design/build_v07.py').read_text(),
        'rgb_reference_illustration': '连续参考线的 RGB 投影' in text and '车体系连续参考线' not in text,
        'descriptive_proposal_prose': not re.search(r'必须|不得|不应|应该|应当|建议|表达要求', source.read_text()),
        'no_hidden_source_caption': all(t not in architecture_all_text for t in ('两类初始指令', '图1.', '覆盖工作指令')),
        'two_figure_waypoints': displayed_waypoints is not None and len(re.findall(r'\(\d+,\d+\)',displayed_waypoints.group(1)))==2 and build['displayed_output_source']['figure_waypoints']==[[480,650],[480,490]],
        'archived_global_map_pair_integrity': pair['observations']>1000 and pair['source_database_unchanged'] and all(sha(ROOT/'docs/design/figures/warehouse-assets'/name)==h for name,h in pair['image_hashes'].items()),
        'original_v06_figure_reused': build.get('original_figure_source', {}).get('mode') == 'original vector artwork with image overlays and multiview input',
        'controller_trajectory_only': build.get('original_figure_source', {}).get('controller_input') == 'selected trajectory only' and all(t not in architecture_text for t in ('Current state / obstacles', 'Safety Shield')),
        'functional_architecture': all(t not in architecture_text for t in ('Text tokenizer', 'Vision encoder', 'InternVL3-1B')),
        'fast_slow_mapping_and_landmark_gate': all(t in architecture_text for t in ('HIGH FREQUENCY', 'LOW FREQUENCY', '<landmark>', 'Semantic Annotation', 'Map Fusion', 'Multi-view RGB', 'Scorer')), 
        'v06_unchanged': sha(ROOT / 'dynamic-warehouse-vln-design-v0.6.pdf') == BASELINE,
        'source_matches_build': sha(source) == build['source_sha256'],
        'builder_matches_build': sha(ROOT / 'docs/design/build_v07.py') == build['builder_sha256'],
        'archived_source_matches_build': sha(ROOT / 'docs/design/archive/v0.7-20261028.md') == build['archived_v07_source_sha256'],
        'pdf_matches_build': sha(path) == build['pdf_sha256'],
        'four_parts': [x[1] for x in chapters] == [
            '1. Model Pipeline', '2. Training Schedule',
            '3. Data Engine', '4. Deployment Optimization'],
        'outline_destinations_valid': all(1 <= x[2] <= len(pdf) for x in toc),
        'toc_links_valid': len([x for x in links if x['kind'] == pymupdf.LINK_GOTO]) >= len([x for x in toc if re.match(r'^\d+\.', x[1])])
            and all(0 <= x['page'] < len(pdf) for x in links if x['kind'] == pymupdf.LINK_GOTO),
        'full_row_click_targets': build.get('toc_full_row_links') == len(headings) and len([l for l in links if l['kind'] == pymupdf.LINK_GOTO and l['from'].width > 500]) == len(headings),
        'vlm_grouped': all(x in headings for x in ['1.2.1 输入与递归推理','1.2.2 结构化输出与语义职责','1.2.3 两个图像锚点与连续参考线']),
        'slam_vlm_fast_slow_section': '1.1.2 SLAM 与 VLM 的快慢语义建图' in headings,
        'reconstruction_and_unified_supervision': all(x in headings for x in ['3.1.1 场景采集与度量重建','3.1.2 3DGS 外观与可通行几何','3.2.1 原始数据处理与标准化','3.2.2 教师标注输入输出','3.2.4 标签来源与联合核验']) and not re.search(r'^### 3\.7\s',source.read_text(),re.M),
        'related_architecture_section_removed': '与相关架构的联系' not in source.read_text(),
        'scorer_subsection': '1.5.3 Hydra Scorer：多指标蒸馏与轨迹选择' in headings,
        'hydra_scorer_flow_preserved': all(s in source.read_text() for s in ('L_FM =', 'L_scorer =', '原论文对固定轨迹词表评分', '本方案对 Flow 动态生成的候选评分', 'Stage3：FMAE 与 Hydra Scorer 学习')) and 'Flow Action Expert' in architecture_text and 'Hydra heads' in architecture_text,
        'four_stage_training': all(s in source.read_text() for s in ('Stage1：VLM 语义学习', 'Stage2：Reference Decoder 学习', 'Stage3：FMAE 与 Hydra Scorer 学习', 'Stage4：仓库闭环改进', 'stop_gradient(VLM')) and not re.search(r'Stage1[ab]',source.read_text()) and 'L_semantic =' not in source.read_text(),
        'stage4_joint_adaptation': all(s in source.read_text() for s in ('在联合微调、DAgger 与 RL 中均参与更新', 'VLM 使用文本序列策略梯度', '条件梯度经连续参考线回传 Decoder', 'VLM 更新后')) and "('Stage4','仓库闭环',18,90,('train','train','train')" in (ROOT/'docs/design/build_v07.py').read_text(),
        'reference_error_and_delay_learning': all(s in source.read_text() for s in ('1.4.1 语义引导与静态路线纠错', '1.4.2 延迟对齐与当前观测修正', 'L_trusted_anchor', '30% 干净输入、40% 真实学生预测、30% 合成扰动', '输入错误不改变正确路线标签')),
        'multitask_stage1_recipe': [int(x) for x in re.findall(r'^\| (\d+)% \|', source.read_text().split('#### 2.2.2 ')[1].split('#### 2.2.3 ')[0], re.M)] == [35,20,10,10,5,8,7,5],
        'multitask_stage2_recipe': [int(x) for x in re.findall(r'^\| (\d+)% \|', source.read_text().split('#### 2.3.2 ')[1].split('#### 2.3.3 ')[0], re.M)] == [40,25,20,15],
        'training_grouped_by_stage': re.findall(r'^### (2\..+)$',source.read_text(),re.M)==['2.1 训练路线与参数更新','2.2 Stage1：VLM 语义学习','2.3 Stage2：Reference Decoder 学习','2.4 Stage3：FMAE 与 Hydra Scorer 学习','2.5 Stage4：仓库环境闭环改进'] and all(s in headings for s in ('2.2.2 数据来源与统一配方','2.2.3 消融与规模设计','2.3.2 数据来源与配方','2.3.3 训练评价','2.4.2 数据来源、配方与候选构造','2.4.3 消融设计','2.5.2 数据来源、配方与回放','2.5.3 消融设计')),
        'requested_sections_removed': not re.search(r'^### (?:2\.[67]|4\.[56])\s',source.read_text(),re.M),
        'context_fusion_internal_to_fmae': 'Context Compressor' not in source.read_text() and '上下文压缩器' not in source.read_text() and 'FMAE 内部将同次 VLM 推理的 K/V' in source.read_text(),
        'demo_figures_exported': len(build.get('demo_files',{}))==18 and all(sha(ROOT/p)==h for p,h in build['demo_files'].items()),
        'demo_sources_match': sha(ROOT/'docs/design/demo_figures.py')==build['demo_builder_sha256'] and len(build['demo_assets'])==4 and all(sha(ROOT/p)==h for p,h in build['demo_assets'].items()),
        'reference_figure_in_decoder': source.read_text().index('### 1.4 ') < source.read_text().index('<!-- diagram:anchors -->') < source.read_text().index('### 1.5 '),
        'reference_no_ablation': 'Decoder 消融比较' not in source.read_text() and 'Reference Decoder 采用固定结构与训练配方，不设置消融' in source.read_text(),
        'fmae_two_ablations': len(re.findall(r'^\| FMAE ',source.read_text().split('#### 2.4.3 ')[1].split('### 2.5 ')[0],re.M))==2,
        'ablation_question_settings_metrics': source.read_text().count('| 消融什么 | 设计不同参数实验 | 验证什么 |')==3,
        'training_figure_exported': len(build.get('training_files',{}))==2 and all(sha(ROOT/p)==h for p,h in build.get('training_files',{}).items()),
        'two_static_data_sources': [h for h in headings if h.startswith('3.3.')] == [
            '3.3.1 示教与人工移动数据', '3.3.1.1 未来轨迹截取与几何监督生成', '3.3.1.2 碰撞与监督一致性校验',
            '3.3.2 新任务与地图记忆数据', '3.3.2.1 任务构造与历史地图条件',
            '3.3.2.2 轨迹 Oracle 与仿真示范生成', '3.3.2.3 碰撞与任务一致性校验'],
        'dataengine_figures_match': sha(ROOT/'docs/design/dataengine_figures.py') == build['dataengine_builder_sha256'] and all(sha(ROOT/p)==h for p,h in build['dataengine_assets'].items()),
        'dataengine_causal_example': json.loads((ROOT/'docs/design/figures/dataengine-assets/evidence.json').read_text())['memory']['map_cutoff'] < json.loads((ROOT/'docs/design/figures/dataengine-assets/evidence.json').read_text())['memory']['source_time'],
        'dataengine_two_projected_anchors': len(json.loads((ROOT/'docs/design/figures/dataengine-assets/evidence.json').read_text())['demonstration']['waypoint_uv']) == 2,
        'oracle_design': '何时需要轨迹 Oracle' not in source.read_text() and '3.3.2.2 轨迹 Oracle 与仿真示范生成' in headings,
        'all_numbered_headings_in_printed_toc': all(re.sub(r'\s+', '', h) in toc_text for h in headings),
        'nonempty_pages': all(len(x.strip()) > 100 for x in texts),
        'no_replacement_characters': '\ufffd' not in text,
        'no_text_outside_pages': not overflow,
        'protocol_text_present': all(x in text for x in ('<landmark>', '<local>', '<Route>', '<waypoint>', '<REFLINE>', 'Previous Local:', '(u1,v1);(u2,v2)')),
        'selected_diagrams': build.get('diagrams') == ['overview','mapping','anchors','fmae','training','gsplat','annotation','demonstration','memorytask','navmesh','navmeshquery','deployment'],
        'all_numbered_headings_in_outline': [x[1] for x in toc if re.match(r'^\d+\.',x[1])] == re.findall(r'^#{2,5} (\d+\..+)$',source.read_text(),re.M),
        'illustration_assets_match': all(sha(ROOT/p)==h for p,h in build.get('illustration_assets',{}).items()) and len(build.get('illustration_assets',{}))==7,
        'asset_provenance_matches': len(build.get('asset_provenance',{}))==6 and all(sha(ROOT/p)==h for p,h in build.get('asset_provenance',{}).items()),
        'real_multiview_and_rgb_reference': all(t in architecture_text for t in ('Left','Right','Reference line')) and 'F / L / R / B' not in architecture_text,
        'simplified_module_labels': 'SLAM' in architecture_text and 'Geometry Fusion' not in architecture_text and all(t not in architecture_text for t in ('Depth + calibration','Vision + WS + IMU','Align + associate','Target loss')),
        'frame_output_callouts': all(t in architecture_text for t in ('<landmark>','<local>','<Route>','<waypoint>','03/4R','04/4R','STOP.')) and sha(ROOT/build['displayed_output_source']['file'])==build['displayed_output_source']['sha256'],
        'engineering_section_removed': '代码落点与模型侧验收' not in text,
        'diagram_captions_present': all(re.search(r'图\s*'+str(n)+r'\s*[：:]',text) for n in range(1,13)),
        'revision_date_present': '2026-11-27' in text,
        'all_pages_rendered': all((args.renders / f'page-{n}.png').is_file() for n in range(1, len(pdf) + 1)),
    }
    report = dict(checks=checks, passed=all(checks.values()), pages=len(pdf),
                  chapters=chapters, overflow=overflow, renders=str(args.renders),
                  source_sha256=sha(source), pdf_sha256=sha(path),
                  visual_review='not assessed by automated checks',
                  scope='Document integrity only; no new navigation or model validation.')
    (ROOT / 'docs/design/v0.7-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    build['render_checks'] = dict(passed=report['passed'], pages_rendered=len(pdf),
                                  report='docs/design/v0.7-verification.json')
    (ROOT / 'docs/design/v0.7-build.json').write_text(json.dumps(build, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(checks, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
