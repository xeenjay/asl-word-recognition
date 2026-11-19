#!/usr/bin/env python3
"""Build descriptive validation analysis and a deterministic video gallery.

Reads saved outputs, verifies their binding to the completed comparison audit,
recomputes every metric, and copies existing validation videos byte-for-byte.
Never trains, invokes a model, downloads a dataset, or reads test media.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment')
POET = Path('runs/20260923T050910Z-official-val-v2-evaluation-019ecd54/evaluation')
AUDIT = Path('runs/20261002T180309Z-classical-comparison-analysis-e47f825b/analysis/analysis.json')
CACHE = Path('runs/20260923T050621Z-val-v2-complete-cache-df2e5b93/preprocessing/cache_manifest.json')
FACTORIAL = Path('runs/20260926T224941Z-factorial-retrieval-analysis-06b2704e/analysis.json')
IDS = ['knn', 'linear_svm', 'random_forest', 'poet_official_v2']
NAMES = dict(zip(IDS, ['kNN', 'Linear SVM', 'Random Forest', 'Official PoET']))
CATEGORIES = {
 'all_correct': ('四种方法都正确', 'All four correct'),
 'all_wrong': ('四种方法都错误', 'All four wrong'),
 'classical_rescue': ('PoET 错，传统方法至少一个对', 'PoET wrong, a classical method correct'),
 'poet_only': ('仅 PoET 正确', 'Only PoET correct'),
 'no_hand_detection': ('整段未检测到手部关键点', 'No hand landmarks detected in the clip'),
}
DESCRIPTIONS = {
 'knn': ('把视频压缩为 1020 维姿态/运动描述符，经训练集标准化后，检索 cosine 距离最近的 5 个训练样本并等权投票。', 'Summarize the clip as a 1,020-dimensional pose/motion descriptor, standardize using train statistics, retrieve five cosine-nearest training examples, and vote uniformly.'),
 'linear_svm': ('同一固定描述符上，用 scikit-learn 的 LinearSVC 训练 2,731 个 one-vs-rest 线性判别器；C=0.01。', 'Fit 2,731 one-vs-rest linear classifiers with scikit-learn LinearSVC on the same fixed descriptors; C=0.01.'),
 'random_forest': ('同一固定描述符上，用 scikit-learn 的 64 棵随机树投票，每棵最多 256 个叶子。', 'Fit 64 scikit-learn random trees to the same fixed descriptors, with at most 256 leaves per tree.'),
 'poet_official_v2': ('官方训练权重读取 128 帧 × 188 维关键点序列，使用小型 Transformer 学习时序特征并输出词标签；本表为原版官方权重。', 'The official trained checkpoint consumes a 128-frame by 188-coordinate pose sequence and uses a compact Transformer to predict one word label. This table uses the original official weights.'),
}
LIMITS = {
 'knn': ('距离依赖手工特征，保存完整训练参考库；这里的检索不是改进 PoET 的 learned-embedding 检索。', 'Distance quality depends on hand-engineered features; the full reference bank is retained. This is distinct from the learned-embedding retrieval used in improved PoET.'),
 'linear_svm': ('线性边界建立在时间汇总特征上，不显式建模逐帧注意力；decision margin 不是概率。', 'Linear boundaries act on temporally summarized descriptors and do not model framewise attention; decision margins are not probabilities.'),
 'random_forest': ('本次限制树数量和叶子数以控制内存，结果不代表所有随机森林配置的上限。', 'Tree and leaf counts were deliberately bounded for memory; this is not an upper bound on all random-forest configurations.'),
 'poet_official_v2': ('使用已训练的官方权重及不同的序列表示；与三个固定特征方法的差距不能全部归因于分类器。', 'Uses a trained official checkpoint and a different sequence representation; the gap to the fixed-feature methods cannot be attributed solely to the classifier.'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def local_relative(path):
    text = str(path)
    marker = '/cv-asl-poet/'
    return text.split(marker, 1)[1] if marker in text else text


class Evidence:
    def __init__(self, root):
        self.root = root
        self.inputs = {}

    def read(self, relative, expected=None):
        relative = Path(relative)
        p = self.root / relative
        digest = sha(p)
        require(expected is None or digest == expected, f'Hash mismatch: {relative}')
        self.inputs[str(relative)] = {'path': str(relative), 'sha256': digest, 'size_bytes': p.stat().st_size}
        return p.read_text(encoding='utf-8')

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def recheck(self):
        for item in self.inputs.values():
            require(sha(self.root / item['path']) == item['sha256'], f'Input changed: {item["path"]}')


def metrics(indices, predictions, labels):
    result = {}
    for method, rows in predictions.items():
        top1 = sum(rows[i]['top5_ids'][0] == labels[rows[i]['gloss']] for i in indices)
        top5 = sum(labels[rows[i]['gloss']] in rows[i]['top5_ids'] for i in indices)
        result[method] = {'top1': top1 / len(indices) if indices else None,
                          'top5': top5 / len(indices) if indices else None,
                          'correct': top1, 'top5_correct': top5, 'count': len(indices)}
    return result


def export_frames(source, targets, backend, frame_records, case_id):
    """Export 25/50/75% frames without resizing, then record their hashes."""
    if backend == 'reuse':
        require(all(p.is_file() for _, p in targets), f'Frames unavailable: {case_id}')
        old = frame_records.get(case_id)
        require(old is not None, f'No saved frame provenance: {case_id}')
        for (_, path), saved in zip(targets, old['frames']):
            require(sha(path) == saved['sha256'], f'Frame hash mismatch: {path}')
        require(sha(source) == old['source_sha256'], f'Frame source changed: {case_id}')
        return old
    if backend == 'opencv':
        import cv2
        cap = cv2.VideoCapture(str(source))
        require(cap.isOpened(), f'Cannot decode: {source}')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = {int((count - 1) * f + .5): (f, p) for f, p in targets}
        frame_meta = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in indices:
                fraction, path = indices[index]
                require(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]), 'JPEG write failed')
                frame_meta.append({'fraction': fraction, 'frame_index': index,
                                   'actual_seconds': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                   'file': 'assets/' + path.name, 'sha256': sha(path)})
            index += 1
        cap.release()
        require(index == count and len(frame_meta) == len(targets), 'Frame extraction incomplete')
        return {'tool': 'OpenCV', 'version': cv2.__version__, 'source_sha256': sha(source), 'frames': frame_meta}
    require(backend == 'swift', f'Unknown frame backend {backend}')
    swift = r'''
import Foundation
import AVFoundation
import AppKit
let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
let generator = AVAssetImageGenerator(asset: asset)
generator.appliesPreferredTrackTransform = true
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
let duration = CMTimeGetSeconds(asset.duration)
var rows: [[String: Any]] = []
for j in stride(from: 2, to: CommandLine.arguments.count, by: 2) {
 let fraction = Double(CommandLine.arguments[j])!
 let target = duration * fraction
 var actual = CMTime.zero
 let image = try generator.copyCGImage(at: CMTime(seconds: target, preferredTimescale: 600), actualTime: &actual)
 let bitmap = NSBitmapImageRep(cgImage: image)
 let data = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.92])!
 try data.write(to: URL(fileURLWithPath: CommandLine.arguments[j+1]))
 rows.append(["fraction": fraction, "requested_seconds": target, "actual_seconds": CMTimeGetSeconds(actual)])
}
let data = try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
'''
    with tempfile.TemporaryDirectory(prefix='asl-gallery-') as tmp:
        script = Path(tmp) / 'frames.swift'; script.write_text(swift)
        command = ['swift', '-module-cache-path', str(Path(tempfile.gettempdir()) / 'asl-gallery-swift-cache'), str(script), str(source)]
        for fraction, target in targets:
            command.extend([str(fraction), str(target)])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        require(completed.returncode == 0, 'Swift frame extraction failed: ' + completed.stderr[-1500:])
        rows = json.loads(completed.stdout)
    for row, (_, path) in zip(rows, targets):
        row.update(file='assets/' + path.name, sha256=sha(path))
    return {'tool': 'macOS AVFoundation / Swift', 'strategy': 'Nearest frame at 25%, 50%, 75% of asset duration, zero time tolerance; no crop or resize.', 'source_sha256': sha(source), 'frames': rows}


def build(root, output, frame_backend):
    evidence = Evidence(root)
    audit = evidence.json(AUDIT)
    require(audit['status'] == 'completed' and audit['test_used'] is False, 'Comparison audit not complete')
    bound = {local_relative(x['path']): x['sha256'] for x in audit['inputs']}
    def audited(relative):
        require(str(relative) in bound, f'No historical hash binding: {relative}')
        return evidence.read(relative, bound[str(relative)])
    dataset = json.loads(audited(EXPERIMENT / 'dataset.json'))
    labels_list = dataset['id_to_label']; labels = {word: i for i, word in enumerate(labels_list)}
    require(len(labels) == 2731, 'Unexpected vocabulary')
    predictions = {}
    for method in IDS:
        path = POET / 'predictions.jsonl' if method == IDS[-1] else EXPERIMENT / method / 'predictions.jsonl'
        rows = [json.loads(line) for line in audited(path).splitlines()]
        if method == IDS[-1]:
            rows = [{**row, 'top5_ids': [x['class_id'] for x in row['top5']],
                     'top5_scores': [x['probability'] for x in row['top5']], 'score_kind': 'softmax_probability'} for row in rows]
        require(len(rows) == 10304, f'Incomplete validation: {method}')
        for i, row in enumerate(rows):
            require(row['index'] == i and row['status'] == 'ok' and row['split'] == 'val', f'Invalid row: {method}/{i}')
            require(1 <= len(row['top5_ids']) <= 5 and len(set(row['top5_ids'])) == len(row['top5_ids']), 'Invalid top-k')
            require(all(type(x) is int and 0 <= x < len(labels) for x in row['top5_ids']), 'Unknown prediction class')
            require(len(row['top5_scores']) == len(row['top5_ids']) and all(math.isfinite(x) for x in row['top5_scores']), 'Invalid scores')
            if predictions:
                reference = predictions[IDS[0]][i]
                require(all(row[key] == reference[key] for key in ('index','gloss','signer_id','split','video_sha256','source_path')), f'Identity mismatch: {method}/{i}')
        predictions[method] = rows
    cohort = predictions[IDS[0]]
    require(len({r['video_sha256'] for r in cohort}) == 10304, 'Duplicate validation video')
    all_metrics = metrics(range(10304), predictions, labels)
    methods = []
    for method in IDS:
        rows = predictions[method]; recomputed = all_metrics[method]
        old = next(x for x in audit['comparison'] if x['method'] == method)
        require(recomputed['correct'] == old['top1_correct'] and recomputed['top5_correct'] == old['top5_correct'], 'Prior comparison disagrees')
        by_class = defaultdict(list)
        for i, row in enumerate(rows): by_class[row['gloss']].append(i)
        macro = sum(sum(rows[i]['top5_ids'][0] == labels[word] for i in indices) / len(indices) for word, indices in by_class.items()) / len(by_class)
        m = {'id': method, 'name': NAMES[method], 'top1': recomputed['top1'], 'top5': recomputed['top5'], 'macro_top1': macro,
             'top1_correct': recomputed['correct'], 'top5_correct': recomputed['top5_correct'], 'count': 10304, 'failed': 0,
             'score_type': rows[0]['score_kind'], 'description_zh': DESCRIPTIONS[method][0], 'description_en': DESCRIPTIONS[method][1],
             'limitations_zh': LIMITS[method][0], 'limitations_en': LIMITS[method][1]}
        if method != IDS[-1]:
            details = evidence.json(EXPERIMENT / method / 'method_details.json')
            m['cached_descriptor_timings'] = details['timings']
        methods.append(m)
    correct = {m: [r['top5_ids'][0] == labels[r['gloss']] for r in predictions[m]] for m in IDS}
    pairwise = []
    for a, b in combinations(IDS, 2):
        cc = sum(correct[a][i] and correct[b][i] for i in range(10304))
        ac = sum(correct[a][i] and not correct[b][i] for i in range(10304))
        bc = sum(correct[b][i] and not correct[a][i] for i in range(10304))
        pairwise.append({'method_a': a, 'method_b': b, 'count': 10304, 'both_correct': cc, 'only_a_correct': ac,
                         'only_b_correct': bc, 'both_incorrect': 10304-cc-ac-bc, 'either_correct': cc+ac+bc,
                         'oracle_top1': (cc+ac+bc)/10304})
    signers = []
    for signer in sorted({r['signer_id'] for r in cohort}):
        indices = [i for i, row in enumerate(cohort) if row['signer_id'] == signer]
        signers.append({'signer_id': signer, 'count': len(indices), 'metrics': metrics(indices,predictions,labels)})
    cache = evidence.json(CACHE)
    require(cache['status'] == 'completed' and len(cache['samples']) == 10304, 'Incomplete pose metadata')
    metadata = {}
    for row in cache['samples']:
        index = row['index']; reference = cohort[index]
        require(all(row[k] == reference[k] for k in ('video_sha256','gloss','signer_id','split')), 'Pose metadata identity mismatch')
        v = row['video_metadata']; detections = v['detection_counts']; frames = v['decoded_frames']
        require(frames > 0 and v['fps'] > 0, 'Bad video metadata')
        metadata[index] = {'duration_seconds': frames/v['fps'], 'decoded_frames': frames,
                           'left_hand_fraction': detections['left_hand']/frames,
                           'right_hand_fraction': detections['right_hand']/frames,
                           'mean_hand_detection_fraction': (detections['left_hand']+detections['right_hand'])/(2*frames),
                           'body_detection_fraction': detections['body_pose']/frames}
    groups = []
    definitions = [
      ('duration', 'under_2s', '< 2 秒', '< 2 s', lambda m: m['duration_seconds'] < 2),
      ('duration', '2_to_3s', '2–<3 秒', '2–<3 s', lambda m: 2 <= m['duration_seconds'] < 3),
      ('duration', '3_to_5s', '3–<5 秒', '3–<5 s', lambda m: 3 <= m['duration_seconds'] < 5),
      ('duration', '5s_plus', '≥ 5 秒', '≥ 5 s', lambda m: m['duration_seconds'] >= 5),
      ('hand_detection', 'zero', '双手检测均为 0', 'Zero detections of either hand', lambda m: m['mean_hand_detection_fraction'] == 0),
      ('hand_detection', 'low', '双手平均检测率 (0, 25%)', 'Mean hand detection (0, 25%)', lambda m: 0 < m['mean_hand_detection_fraction'] < .25),
      ('hand_detection', 'medium', '双手平均检测率 [25%, 50%)', 'Mean hand detection [25%, 50%)', lambda m: .25 <= m['mean_hand_detection_fraction'] < .5),
      ('hand_detection', 'high', '双手平均检测率 ≥ 50%', 'Mean hand detection ≥ 50%', lambda m: m['mean_hand_detection_fraction'] >= .5),
    ]
    for dimension, gid, zh, en, predicate in definitions:
        indices = [i for i in range(10304) if predicate(metadata[i])]
        groups.append({'dimension': dimension, 'id': gid, 'label_zh': zh, 'label_en': en,
                       'count': len(indices), 'metrics': metrics(indices,predictions,labels)})
    for dimension in ('duration','hand_detection'):
        require(sum(x['count'] for x in groups if x['dimension']==dimension) == 10304, 'Groups do not partition cohort')
    categories = {
      'all_correct': [i for i in range(10304) if all(correct[m][i] for m in IDS)],
      'all_wrong': [i for i in range(10304) if not any(correct[m][i] for m in IDS)],
      'classical_rescue': [i for i in range(10304) if not correct[IDS[-1]][i] and any(correct[m][i] for m in IDS[:-1])],
      'poet_only': [i for i in range(10304) if correct[IDS[-1]][i] and not any(correct[m][i] for m in IDS[:-1])],
      'no_hand_detection': [i for i in range(10304) if metadata[i]['mean_hand_detection_fraction']==0],
    }
    quotas = {'all_correct':4,'all_wrong':4,'classical_rescue':4,'poet_only':2,'no_hand_detection':2}
    output.mkdir(parents=True, exist_ok=True); (output/'assets').mkdir(exist_ok=True)
    frame_path = output / 'frame-provenance.json'
    prior_frames = json.loads(frame_path.read_text()) if frame_path.exists() else {}
    frame_records = {}; chosen = set(); cases = []
    for category, quota in quotas.items():
        available = [i for i in categories[category] if i not in chosen][:quota]
        require(len(available)==quota, f'Category too small: {category}')
        for index in available:
            chosen.add(index); row = cohort[index]; case_id = f'val-{index:05d}'
            original = Path(row['source_path']).name
            source = root/'data/asl_citizen/videos'/original
            require(sha(source) == row['video_sha256'], f'Selected media mismatch: {index}')
            video_file = f'assets/{case_id}.mp4'; shutil.copyfile(source, output/video_file)
            targets = [(f, output/'assets'/f'{case_id}-{int(100*f)}.jpg') for f in (.25,.5,.75)]
            record = export_frames(source, targets, frame_backend, prior_frames, case_id)
            frame_records[case_id] = record
            top5 = {m: [{'gloss':labels_list[g], 'score':s, 'rank':rank+1} for rank,(g,s) in enumerate(zip(predictions[m][index]['top5_ids'],predictions[m][index]['top5_scores']))] for m in IDS}
            truths = {m:correct[m][index] for m in IDS}
            poet_rank = next((i+1 for i,x in enumerate(top5[IDS[-1]]) if x['gloss']==row['gloss']), None)
            observations_zh = f'标签为 {row["gloss"]}；四方法 Top-1 正确数 {sum(truths.values())}/4。PoET 的正确标签' + (f'在第 {poet_rank} 名。' if poet_rank else '不在 Top-5 中。')
            observations_en = f'Ground truth: {row["gloss"]}; {sum(truths.values())}/4 methods have correct Top-1. The target ' + (f'is ranked {poet_rank} by PoET.' if poet_rank else 'is absent from PoET Top-5.')
            if category == 'no_hand_detection':
                hypothesis_zh = '缓存记录整段双手检测数均为 0，但这不等于画面没有手；还保留身体/面部信息。需要检查视频与关键点覆盖才能判断检测问题，不能仅由错误归因。'
                hypothesis_en = 'The cache has zero detections of either hand, which does not mean no hands are visible. Body/face information remains. Inspect video and landmark coverage before attributing an error to the detector.'
            elif category == 'classical_rescue':
                hypothesis_zh = '传统描述符与 learned sequence representation 可能保留不同区分信息；单例说明互补，不证明按置信度融合会更好。'
                hypothesis_en = 'Hand-engineered descriptors and learned sequence representations may retain different cues. This case shows complementarity, not that confidence-based fusion would improve accuracy.'
            elif category == 'all_wrong':
                hypothesis_zh = '可继续检查时序、手形细节与关键点缺失；这里尚无受控反事实，不能认定具体失败原因或标签错误。'
                hypothesis_en = 'Temporal detail, handshape, and landmark loss are candidates for further inspection. No controlled counterfactual here establishes a failure cause or a labeling error.'
            elif category == 'poet_only':
                hypothesis_zh = 'PoET 可能受益于可学习时序表示，但特征表示和训练来源同时不同；不能据单例把优势完全归于 Transformer。'
                hypothesis_en = 'PoET may benefit from its learned temporal representation, but representation and training provenance both differ; this case cannot isolate the effect of a Transformer.'
            else:
                hypothesis_zh = '此例用于理解四种输出，不代表四种方法在其他词或新用户视频上都可靠。'
                hypothesis_en = 'Use this example to understand the outputs; success does not establish reliability for other words or new-user videos.'
            cases.append({'id':case_id,'index':index,'gloss':row['gloss'],'signer_id':row['signer_id'],
                          'category':category,'category_zh':CATEGORIES[category][0],'category_en':CATEGORIES[category][1],
                          'video_file':video_file,'poster_file':targets[1][1].relative_to(output).as_posix(),
                          'frame_files':[p.relative_to(output).as_posix() for _,p in targets],
                          'video_sha256':row['video_sha256'],'original_filename':original,
                          'metadata':metadata[index], 'top5':top5,'correct':truths,
                          'score_types':{m:predictions[m][index]['score_kind'] for m in IDS},
                          'observations_zh':observations_zh,'observations_en':observations_en,
                          'hypothesis_zh':hypothesis_zh,'hypothesis_en':hypothesis_en,
                          'selection_rule':f'First {quota} unused validation indices satisfying {category}, ascending index; categories processed in documented order.'})
    dump(frame_path,frame_records)
    factorial_source = evidence.json(FACTORIAL, '7d919e7017852e419e5992aef91abdcfa5c49320722d74c3a7671c864205d1a5')
    require(factorial_source['status']=='verified' and factorial_source['test_used'] is False, 'Factorial audit not verified')
    f = factorial_source['factorial']; summary=[]; per_seed=[]
    for cell, name in [('B','Fine-tuned PoET'),('B_plus_R','PoET + retrieval'),('G','PoET + geometry'),('G_plus_R','PoET + geometry + retrieval')]:
        cell_data=f['aggregate']['cells'][cell]
        summary.append({'id':cell,'name':name,'mean_top1':cell_data['top1']['mean'],'std_top1':cell_data['top1']['sample_std'],
                        'mean_macro_top1':cell_data['macro_top1']['mean'],'std_macro_top1':cell_data['macro_top1']['sample_std']})
    for seed, data in f['by_seed'].items():
        s={'seed':int(seed)}
        for cell, cell_data in data['cells'].items():
            s[cell]={'top1':cell_data['top1'],'macro_top1':cell_data['macro_top1'],'top1_correct':cell_data['correct_top1']}
        comparison=data['comparisons']['G_plus_R_vs_B']
        s.update({k:comparison[k] for k in ('fixed_count','regressed_count','net_correct')})
        s['delta_pp']=comparison['delta_top1_percentage_points'];per_seed.append(s)
    factorial={'summary':summary,'per_seed':per_seed,'delta_pp':f['aggregate']['comparisons']['G_plus_R_vs_B']['delta_top1_percentage_points']['mean'],
               'interaction_pp':f['aggregate']['interaction']['top1_percentage_points']['mean'],
               'notes_zh':'B 是匹配预算的微调 PoET，不是上表官方权重。三 seed 共用反复开发的 validation；均值±样本 SD 不证明显著或独立泛化。R 只重排原 Top-5，Top-5 命中不变；检索使用 learned embeddings，区别于传统 kNN。',
               'notes_en':'B is the budget-matched fine-tuned PoET, not the official checkpoint in the four-method table. All three seeds share a repeatedly used development validation set; mean ± sample SD does not establish significance or independent generalization. R only reranks the original Top-5, so its Top-5 hits are unchanged; retrieval uses learned embeddings, unlike the classical kNN.'}
    # Fixed representative seed; read actual matched predictions, not official PoET.
    rerank_run = Path('runs/20260926T224853Z-factorial-retrieval-rerank-94b64b9e')
    artifact_file = rerank_run / 'artifacts.json'
    artifacts = evidence.json(artifact_file, factorial_source['input_sha256'][str(artifact_file)])
    artifact_hashes = {item['path']: item['sha256'] for item in artifacts['files']}
    matched = {}
    for cell, model_id, method in [('B', 'baseline_seed20260923', 'original'),
                                    ('G_plus_R', 'geometry_seed20260923', 'fixed_blend')]:
        relative = Path('rerank/models') / model_id / method / 'predictions.jsonl'
        raw = evidence.read(rerank_run / relative, artifact_hashes[str(relative)])
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == 10304, f'Incomplete matched predictions: {cell}')
        for index, row in enumerate(rows):
            reference = cohort[index]
            require(row['index'] == index and row['row'] == index and row['model_id'] == model_id
                    and row['method'] == method and row['split'] == 'val', 'Wrong matched model/row')
            require(all(row[k] == reference[k] for k in ('gloss', 'signer_id', 'video_sha256')), 'Matched video identity mismatch')
            require(row['identity'] == {'gloss': reference['gloss'], 'signer_id': reference['signer_id'],
                    'sha256': reference['video_sha256'], 'split': 'val',
                    'path': 'videos/' + Path(reference['source_path']).name}, 'Matched nested identity mismatch')
            require(row['target'] == labels[row['gloss']] and row['prediction'] == row['ranked_class_ids'][0],
                    'Matched target/ranking mismatch')
            require(row['correct_top1'] == (row['prediction'] == row['target']), 'Matched correctness mismatch')
        matched[cell] = rows
    paired = {'fixed': [i for i in range(10304) if not matched['B'][i]['correct_top1'] and matched['G_plus_R'][i]['correct_top1']],
              'regressed': [i for i in range(10304) if matched['B'][i]['correct_top1'] and not matched['G_plus_R'][i]['correct_top1']]}
    historical = f['by_seed']['20260923']['comparisons']['G_plus_R_vs_B']
    factorial_cases = []
    for category, indices in paired.items():
        require(len(indices) == historical[category + '_count']
                and indices == sorted(row['index'] for row in historical[category]), 'Matched paired audit mismatch')
        for index in indices[:2]:
            case = {'seed': 20260923, 'category': category, 'index': index,
                    'gloss': cohort[index]['gloss'], 'signer_id': cohort[index]['signer_id'],
                    'video_sha256': cohort[index]['video_sha256'],
                    'selection_rule': f'First two ascending validation indices with {category} Top-1 relative to matched B, seed 20260923; illustrative, not random.'}
            for cell, rows in matched.items():
                row = rows[index]
                case[cell] = {'prediction': labels_list[row['prediction']], 'correct': row['correct_top1'],
                              'model_id': row['model_id'], 'checkpoint_sha256': row['checkpoint_sha256']}
            factorial_cases.append(case)
    factorial['cases'] = factorial_cases
    factorial['case_selection'] = {'seed': 20260923, 'fixed_count': len(paired['fixed']),
                                  'regressed_count': len(paired['regressed']), 'order': 'fixed then regressed; ascending validation index',
                                  'per_category': 2, 'new_media_exported': False}
    confusions=[]
    for method, rows in predictions.items():
        counts=Counter((row['gloss'],labels_list[row['top5_ids'][0]]) for row in rows if row['gloss']!=labels_list[row['top5_ids'][0]])
        ordered=sorted(counts.items(),key=lambda item:(-item[1],item[0][0],item[0][1]))[:12]
        confusions.extend({'method':method,'true_gloss':a,'predicted_gloss':b,'count':count} for (a,b),count in ordered)
    for note in ('FORMAL_TRAINING_RESULT.md', 'GROUPED_LR_RESULT.md',
                 'CROSS_SIGNER_RESULT.md', 'HAND_RGB_RESULT.md', 'RETRIEVAL_PILOT_RESULT.md'):
        evidence.read(Path('docs') / note)
    evidence.read(Path('scripts/build_comparison_gallery.py'))
    evidence.recheck()
    result={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'verified',
            'split':'val','count':10304,'num_classes':2731,'test_used':False,'new_model_inference':False,
            'methods':methods,'pairwise':pairwise,'per_signer':signers,'groups':groups,'cases':cases,'factorial':factorial,
            'confusions':confusions,
            'category_counts':{k:len(v) for k,v in categories.items()},
            'provenance':{'inputs':sorted(evidence.inputs.values(),key=lambda x:x['path']),
               'selection_policy':{'order':list(quotas),'quotas':quotas,'sort':'ascending project validation index','duplicates':'skip already selected indices','representative':False,'label_based_selection':True},
               'scope_notes_zh':'所有总体指标使用完整 10,304 条 validation。16 个例子是按预测正确性/检测状态挑选的教学样例，不是随机抽样，其比例不能用于估计准确率。重复开发 validation；test 未使用。单手词、起止静止帧和拍摄方式都会影响双手平均检测率，该统计不等于关键点质量或缺失原因。',
               'scope_notes_en':'All aggregate metrics use all 10,304 validation clips. The 16 examples are deliberately selected by prediction correctness/detection status, not randomly sampled; their proportions do not estimate accuracy. Validation has been repeatedly used for development; test is untouched. One-handed signs, idle frames, and recording conditions can all affect mean hand detection, which is not a direct measure of landmark quality or missingness causes.'}}
    review_path=output/'visual-review.json'
    if review_path.exists():
        reviews=json.loads(review_path.read_text()); result['visual_review']=reviews.get('scope')
        for case in cases:
            if case['id'] in reviews.get('cases',{}):
                item=reviews['cases'][case['id']]
                require(item['video_sha256']==case['video_sha256'],'Review media mismatch')
                for frame in item['reviewed_frames']:
                    require(sha(output/frame['file'])==frame['sha256'],'Reviewed frame changed')
                case['visual_observation_zh']=item['observation_zh'];case['visual_observation_en']=item['observation_en']
                case['visual_review_scope']='25%, 50%, 75% extracted frames; no expert ASL transcription'
        result['provenance']['visual_review_sha256']=sha(review_path)
    result['provenance']['frame_provenance_sha256']=sha(frame_path)
    dump(output/'analysis.json',result)
    (output/'data.js').write_text('window.COMPARISON_DATA = '+json.dumps(result,ensure_ascii=False,allow_nan=False).replace('</','<\\/')+';\n',encoding='utf-8')
    write_report(output,result)
    print(json.dumps({'status':'verified','validation_count':10304,'methods':len(methods),'cases':len(cases),
                      'category_counts':result['category_counts'],'output':str(output),'new_inference':False,'test_used':False},ensure_ascii=False))
    return result


def write_report(output, data):
    lines=['# ASL Citizen：方法比较与错误分析 / Methods and error analysis','',
      '所有数值来自已保存并核验的完整 validation 预测；本次只重新计算统计、提取视频展示帧。没有训练、模型推理或 test 评价。',
      'All results below are recomputed from verified saved validation predictions. This build only analyzes outputs and extracts illustration frames; it does not train, run models, or evaluate test.', '',
      '## 任务和比较边界 / Task and comparison scope','',
      '一段预先分好的 ASL 词级视频 → 2,731 个标签之一。训练 40,154 条，validation 10,304 条；官方按 signer 划分。不是连续句子翻译。',
      'One presegmented ASL word clip → one of 2,731 labels. Train: 40,154 clips; validation: 10,304, with signer-disjoint official splits. This is not continuous sentence translation.','',
      '三个传统方法共享 1020 维手工描述符：1/2/4 段时序均值 686 维、全局标准差 98、带符号运动 98、绝对运动 98、覆盖率 40。StandardScaler 只用 train 拟合。MediaPipe 也是已训练的前端，并非从零开发检测器。',
      'The three classical methods share 1,020-dimensional descriptors: 686 temporal-pyramid means (1/2/4 segments), 98 standard deviations, 98 signed-motion and 98 absolute-motion statistics, and 40 coverage features. StandardScaler is fitted on train only; MediaPipe is a pretrained front end.','',
      'PoET 使用官方已有权重和不同的序列表示，传统方法使用本项目固定特征及有限配置。该表是同一任务下的系统比较，不是“只换分类器”的受控实验，也不是所有 kNN/SVM/RF 实现的最佳成绩。',
      'PoET uses official weights and a different sequential representation; classical methods use this project’s descriptors and bounded configurations. This is a system comparison on one task, not a controlled classifier-only ablation or the best possible result for each algorithm.','',
      '| 方法 / Method | Top-1 | Top-5 | Macro Top-1 | 正确数 / Correct |','|---|---:|---:|---:|---:|']
    for m in data['methods']: lines.append(f'| {m["name"]} | {100*m["top1"]:.4f}% | {100*m["top5"]:.4f}% | {100*m["macro_top1"]:.4f}% | {m["top1_correct"]}/10304 |')
    lines += ['','Macro 对词类等权；Top-1/Top-5 对视频等权。所有 2,731 类在 validation 中有样本，0 处理失败。',
      'Macro averages classes equally; Top-1/Top-5 averages clips equally. All 2,731 classes occur in validation; processing failures: zero.','',
      '## 方法解释 / What each method does','']
    for m in data['methods']:
        lines += [f'### {m["name"]}','',m['description_zh'],m['description_en'],'',m['limitations_zh'],m['limitations_en'],'']
    lines += ['SVM/RF 调用 scikit-learn；kNN 的 cosine 搜索与投票由本项目用 NumPy/PyTorch 实现，仍是标准算法。主要新增工作是共享视频特征、数据/词表适配、统一评价和可追溯实验，而非发明这三种算法。',
      'SVM/RF call scikit-learn; the project implements standard cosine search/voting for kNN in NumPy/PyTorch. The project contribution is shared video representation, data/vocabulary adaptation, consistent evaluation, and auditable experiments, not invention of these algorithms.','',
      '**分数不横向比较。** kNN 是邻居票数比例，SVM 是 signed decision margin，RF 是树的平均类别概率，PoET 是 softmax 概率。都未做校准；不能拿不同方法的分数取最大来当有效融合。kNN 少于 5 个类别获得票时 Top-5 列表可能不足 5 个。',
      '**Scores are not cross-model confidence measures.** kNN reports neighbor vote fractions, SVM reports signed decision margins, RF averages tree probabilities, and PoET reports softmax probabilities. None were calibrated; taking the maximum score across methods is not a validated ensemble. kNN may return fewer than five labels when fewer than five classes receive neighbor votes.','',
      '## 错误重叠 / Error overlap','',
      '| A | B | 都对 / Both correct | 仅 A 对 | 仅 B 对 | 都错 / Both wrong |','|---|---|---:|---:|---:|---:|']
    for p in data['pairwise']: lines.append(f'| {NAMES[p["method_a"]]} | {NAMES[p["method_b"]]} | {p["both_correct"]} | {p["only_a_correct"]} | {p["only_b_correct"]} | {p["both_incorrect"]} |')
    sv=next(p for p in data['pairwise'] if p['method_a']=='linear_svm' and p['method_b']==IDS[-1])
    lines += ['',f'SVM 能识别 PoET 错的 {sv["only_a_correct"]} 条，说明错误并非完全重叠；反过来 PoET 对、SVM 错有 {sv["only_b_correct"]} 条。“至少一个正确”的 {sv["either_correct"]}/10304 = {100*sv["oracle_top1"]:.4f}% 需要知道真实标签才能选择，是 oracle 上界诊断，不是已实现融合成绩。',
      f'SVM correctly identifies {sv["only_a_correct"]} clips missed by PoET; PoET correctly identifies {sv["only_b_correct"]} missed by SVM. The either-correct value ({100*sv["oracle_top1"]:.4f}%) requires an oracle to choose the right model and is not an implemented ensemble result.','',
      '## 按 signer 和观测条件分组 / Signer and observational slices','',
      '| Signer | n | kNN | SVM | RF | PoET |','|---|---:|---:|---:|---:|---:|']
    for s in data['per_signer']: lines.append('| '+s['signer_id']+' | '+str(s['count'])+' | '+' | '.join(f'{100*s["metrics"][m]["top1"]:.2f}%' for m in IDS)+' |')
    lines += ['','P13 只有 2 条，百分比非常不稳定。signer 差异同时混杂词类、拍摄条件和表现方式，不能把差异解释为某位参与者能力或某个视觉因素的因果效应。',
      'P13 contains only two clips, so its percentage is unstable. Signer slices mix vocabulary, capture conditions, and signing variation; differences do not measure a participant’s ability or identify the causal effect of one visual factor.','',
      '| 分组 / Group | n | kNN | SVM | RF | PoET |','|---|---:|---:|---:|---:|---:|']
    for s in data['groups']: lines.append('| '+s['label_zh']+' / '+s['label_en']+' | '+str(s['count'])+' | '+' | '.join(f'{100*s["metrics"][m]["top1"]:.2f}%' if s['metrics'][m]['top1'] is not None else 'N/A' for m in IDS)+' |')
    lines += ['','检测率定义为 `(left_hand_count + right_hand_count) / (2 × decoded_frames)`，不是“至少一只手可见的帧比例”；两者不能从边际计数互推。原始持续时间为 `decoded_frames/fps`，不等于实际做手势的有效时长。',
      'Detection fraction is `(left_hand_count + right_hand_count) / (2 × decoded_frames)`, not the fraction of frames with at least one visible hand; marginal counts cannot recover that union. Duration is `decoded_frames/fps`, not active signing duration.','',
      '单手词、起止静止帧、signer/词类分布会影响这些分组。分组结果只描述相关性；没有人为遮挡或匹配词类的对照，不能证明检测缺失、视频较长导致错误。',
      'One-handed signs, idle frames, and signer/class mix affect these slices. These are descriptive associations; no controlled occlusion or class-matched counterfactual establishes that missing landmarks or longer clips caused an error.','',
      '## 16 个真实视频例子 / Sixteen real clips','',data['provenance']['scope_notes_zh'],data['provenance']['scope_notes_en'],'',
      '固定顺序：全对前4、全错前4、PoET错但传统对前4、仅PoET对前2、双手整段未检出前2；每类按项目索引升序，跳过已选。项目索引不是官方 CSV 行号。',
      'Fixed order: first 4 all-correct, first 4 all-wrong, first 4 classical-rescue, first 2 PoET-only, then first 2 no-hand-detection clips. Sort by project index, skipping selected indices; this is not the original CSV row number.','']
    for c in data['cases']:
        lines += [f'### {c["id"]} · {c["gloss"]} · {c["category_zh"]} / {c["category_en"]}','',
          f'[Video]({c["video_file"]}) · [25% frame]({c["frame_files"][0]}) · [50% frame]({c["frame_files"][1]}) · [75% frame]({c["frame_files"][2]})','',c['observations_zh'],c['observations_en'],'',
          '| Method | Top-1 | Correct |','|---|---|---|']
        lines += [f'| {NAMES[m]} | {c["top5"][m][0]["gloss"]} | {c["correct"][m]} |' for m in IDS]
        if c.get('visual_observation_zh'): lines += ['', '**帧检查 / Frame observation:** '+c['visual_observation_zh'],c['visual_observation_en']]
        lines += ['','**解释边界/待验证假设 / Interpretation / hypothesis:** '+c['hypothesis_zh'],c['hypothesis_en'],'']
    f=data['factorial']
    lines += ['## PoET 改进：独立的匹配对照 / Separate matched PoET ablation','',f['notes_zh'],f['notes_en'],'',
      '| Method | Mean Top-1 ± sample SD |','|---|---:|']
    for r in f['summary']: lines.append(f'| {r["name"]} | {100*r["mean_top1"]:.4f}% ± {100*r["std_top1"]:.4f} |')
    lines += ['','| Seed | B correct | G+R correct | Fixed | Regressed | Net | Δ pp |','|---|---:|---:|---:|---:|---:|---:|']
    for r in f['per_seed']:lines.append(f'| {r["seed"]} | {r["B"]["top1_correct"]} | {r["G_plus_R"]["top1_correct"]} | {r["fixed_count"]} | {r["regressed_count"]} | +{r["net_correct"]} | +{r["delta_pp"]:.4f} |')
    lines += ['',f'G+R 平均 +{f["delta_pp"]:.4f} 个百分点；有改对也有改错。Interaction 平均 {f["interaction_pp"]:.4f} 个百分点，不支持“超加成”说法。',
      f'G+R improves by {f["delta_pp"]:.4f} percentage points on average, with both fixes and regressions. Mean interaction is {f["interaction_pp"]:.4f} points, so the evidence does not support super-additivity.','',
      '### 改对与改错的具体例子 / Concrete fixes and regressions','',
      '固定展示 seed20260923：相对匹配预算微调 B，G+R 改对 111 条、改错 41 条，净 +70。下面各取该类别索引最小的两条，双方逐条 video SHA、标签和 signer 身份已验证一致。这里的 B 不是上方四方法比较中的官方 PoET。',
      'For the fixed representative seed 20260923, G+R fixes 111 errors and regresses 41 correct predictions relative to budget-matched fine-tuned B, for a net +70. Each category below shows its first two validation indices. Video SHA, ground truth, and signer identities are matched across both predictions. B here is not the official PoET used in the four-method table.','',
      '| 类型 / Change | Index | 真值 / Target | B Top-1 | G+R Top-1 |','|---|---:|---|---|---|']
    for c in f['cases']:
        change = '改对 / Fixed' if c['category'] == 'fixed' else '改错 / Regressed'
        lines.append(f'| {change} | {c["index"]} | {c["gloss"]} | {c["B"]["prediction"]} | {c["G_plus_R"]["prediction"]} |')
    lines += ['','这些例子解释“净提升”是改对减去改错，而不是所有样本都改善；不从词名推断手语视觉近似或具体失败原因。没有为这四条新增视频资产或模型推理。',
      'These examples explain why a net gain is fixes minus regressions, not improvement on every sample. Label names alone do not establish visual sign similarity or failure causes. No extra media or model inference was produced for these four rows.','',
      '### 负结果与有限收益 / Negative and limited results','',
      '- ASL-LEX 属性辅助头：首轮同 seed 与预算相对 B 只多 4 条；增大新头学习率的后续结果相对 B 只多 2 条。不能把辅助任务 loss 降低当识别显著提升。见 [初轮](../FORMAL_TRAINING_RESULT.md) 与 [分组学习率](../GROUPED_LR_RESULT.md)。',
      '- ASL-LEX auxiliary supervision: only +4 clips initially and +2 with a larger new-head learning rate against the matched baseline. A better auxiliary loss is not itself recognition improvement.',
      '- 跨 signer SupCon：匹配采样器的 CE 为 81.1141%，SupCon 为 81.0850%，净少 3 条（改对 123、改错 126），macro 仅 +0.0238 个百分点。不能把训练期对比 loss 下降等同于识别提高，也不能由一次配置否定所有对比学习。见 [SupCon 对照](../CROSS_SIGNER_RESULT.md)。',
      '- Cross-signer SupCon: matched-sampler CE scores 81.1141% versus 81.0850% for SupCon, a net loss of three clips (123 fixes, 126 regressions), with only +0.0238 points in macro accuracy. Lower contrastive training loss is not recognition improvement; one configuration does not invalidate all contrastive learning.',
      '- RGB 时序分支：同 seed G+RGB GRU 为 81.8517%，旧 G+R 已有 81.8323%，仅净多 2 条；G 的 Top-5 从 96.5450% 降至 96.1277%。新增模态/计算没有换来明显突破。见 [RGB完整结果](../HAND_RGB_RESULT.md)。',
      '- RGB temporal branch: G+RGB GRU reaches 81.8517%, only two clips above same-seed G+R at 81.8323%; G Top-5 drops from 96.5450% to 96.1277%. Extra modality/computation did not deliver a clear breakthrough.',
      '- 官方权重检索 pilot 净多 48 条，未达到事前设定的至少 52 条保留门槛；保留这条负结果，不用后续三 seed 结果改写。见 [retrieval pilot](../RETRIEVAL_PILOT_RESULT.md)。',
      '- The official-checkpoint retrieval pilot gained 48 clips but missed its preregistered 52-clip retention threshold. Later factorial results do not retroactively change that outcome.','',
      '## 能说与不能说 / Defensible conclusions','',
      '可以说：在完整 ASL Citizen validation 上，当前 PoET 系统优于这三种固定特征基线；传统方法之间以 SVM 最好；不同方法仍有互补错误；相对几何与固定检索在匹配三 seed 中提供小幅收益。',
      'Supported: On the complete ASL Citizen validation cohort, the current PoET system outperforms these three fixed-feature baselines; SVM is strongest among the classical methods; errors show complementarity; relative geometry and fixed retrieval give small gains across three matched seeds.','',
      '不能说：证明所有深度方法优于所有传统方法；已经完成连续翻译；这些分数可用于任意手机视频；oracle 是真实融合准确率；0.55 pp 是 test 或显著收益；仅由帧截图诊断了手语错误的真实原因。',
      'Not established: universal deep-learning superiority, continuous translation, these accuracies on arbitrary phone videos, a deployable oracle ensemble, test/statistically significant +0.55 pp, or causal/linguistic diagnosis from sampled stills.','',
      '## 重建 / Rebuild','',
      '在工作区根运行（已有媒体帧重用，验证源视频和帧 SHA）：','```bash','python3 scripts/build_comparison_gallery.py --frames reuse','```',
      '首次在 Linux 的项目环境可用 `--frames opencv`；macOS 可用已有 Swift/AVFoundation 的 `--frames swift`。不安装依赖，不下载数据。完整输入哈希见 `analysis.json.provenance.inputs`，帧来源见 `frame-provenance.json`，视觉观察见 `visual-review.json`。',
      'Use `--frames opencv` for initial frame export in the project’s Linux environment, or `--frames swift` on macOS with existing Swift/AVFoundation. No dependencies or data are downloaded. Input hashes, frame provenance, and visual observations are retained separately.','']
    (output/'METHODS_AND_ERRORS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--frames',choices=['reuse','opencv','swift'],default='reuse')
    args=p.parse_args()
    root=args.root.resolve();output=(args.output or root/'docs/comparison-demo').resolve()
    build(root,output,args.frames)


if __name__=='__main__':
    main()
