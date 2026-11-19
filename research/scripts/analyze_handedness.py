#!/usr/bin/env python3
"""Independent analysis of the handedness experiment (standard library only).

Recomputes overall, mirrored-signer and per-signer metrics from saved prediction
files for the baseline, Arm 0 (query-only) and Arm A (canonical refit), paired
fixed/regressed counts with exact two-sided McNemar p-values, and the verdict
defined in the preregistration. No model is loaded and no test data is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path

METHODS = ('knn', 'linear_svm', 'random_forest')
IDENTITY_KEYS = ('index', 'gloss', 'signer_id', 'split', 'source_path', 'video_sha256')
WORKSPACE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mcnemar_exact(fixed: int, regressed: int) -> float:
    n = fixed + regressed
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(fixed, regressed) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_predictions(path: Path, rows: list, label_ids: dict) -> list[tuple[bool, bool]]:
    lines = path.read_text().splitlines()
    if len(lines) != len(rows):
        raise ValueError(f'{path}: {len(lines)} predictions for {len(rows)} rows')
    outcome = []
    for line, row in zip(lines, rows):
        item = json.loads(line)
        if any(item[k] != row[k] for k in IDENTITY_KEYS) or item.get('status') != 'ok':
            raise ValueError(f'{path}: identity/order/status mismatch at index {row["index"]}')
        target = label_ids[row['gloss']]
        outcome.append((item['top5_ids'][0] == target, target in item['top5_ids']))
    return outcome


def summarize(outcome, rows, mask=None):
    selected = [i for i in range(len(rows)) if mask is None or mask[i]]
    n = len(selected)
    top1 = sum(outcome[i][0] for i in selected)
    top5 = sum(outcome[i][1] for i in selected)
    per_class = {}
    for i in selected:
        entry = per_class.setdefault(rows[i]['gloss'], [0, 0])
        entry[0] += 1
        entry[1] += outcome[i][0]
    macro = sum(c / t for t, c in per_class.values()) / len(per_class) if per_class else None
    return {'count': n, 'top1_correct': top1, 'top1': top1 / n if n else None,
            'top5_correct': top5, 'top5': top5 / n if n else None,
            'macro_top1_observed_classes': macro, 'classes': len(per_class)}


def paired(base, other, mask=None):
    idx = [i for i in range(len(base)) if mask is None or mask[i]]
    fixed = sum((not base[i][0]) and other[i][0] for i in idx)
    regressed = sum(base[i][0] and (not other[i][0]) for i in idx)
    return {'fixed': fixed, 'regressed': regressed, 'net': fixed - regressed,
            'delta_top1_pp': 100 * (fixed - regressed) / len(idx) if idx else None,
            'mcnemar_exact_p': mcnemar_exact(fixed, regressed)}


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    features = args.feature_dir.resolve()
    rows = json.loads((features / 'val_rows.json').read_text())
    document = json.loads((features / 'dataset.json').read_text())
    labels = document['id_to_label']
    label_ids = {label: i for i, label in enumerate(labels)}
    decisions = json.loads((features / 'handedness_decisions.json').read_text())['decisions']
    mirrored = [bool(decisions['val'][r['signer_id']]['mirror']) for r in rows]
    unmirrored = [not m for m in mirrored]
    signers = sorted({r['signer_id'] for r in rows})
    arms = {'baseline': args.baseline_experiment.resolve(), 'arm0_query_only': args.arm0_dir.resolve(),
            'armA_canonical_refit': args.arma_experiment.resolve()}
    inputs = {'preregistration': {'path': str(args.preregistration.resolve()), 'sha256': sha256(args.preregistration)},
              'feature_dataset': {'path': str(features / 'dataset.json'), 'sha256': sha256(features / 'dataset.json')}}
    results, checks = {}, {}
    for method in METHODS:
        outcomes = {}
        for arm, root in arms.items():
            path = root / method / 'predictions.jsonl'
            inputs[f'{arm}/{method}'] = {'path': str(path), 'sha256': sha256(path)}
            outcomes[arm] = load_predictions(path, rows, label_ids)
            saved = json.loads((root / method / 'metrics.json').read_text())
            recomputed = summarize(outcomes[arm], rows)
            if (saved['top1_correct'] != recomputed['top1_correct'] or saved['top5_correct'] != recomputed['top5_correct']):
                raise ValueError(f'{arm}/{method}: saved metrics disagree with recomputation')
        if any(outcomes['arm0_query_only'][i] != outcomes['baseline'][i] for i in range(len(rows)) if unmirrored[i]):
            raise ValueError(f'{method}: Arm 0 changed an unmirrored row')
        checks[method] = {'saved_metrics_recomputed': True, 'arm0_unmirrored_rows_identical_outcomes': True}
        entry = {}
        for arm, outcome in outcomes.items():
            entry[arm] = {'overall': summarize(outcome, rows),
                          'mirrored_signers': summarize(outcome, rows, mirrored),
                          'unmirrored_signers': summarize(outcome, rows, unmirrored),
                          'per_signer': {s: summarize(outcome, rows, [r['signer_id'] == s for r in rows])
                                         for s in signers}}
            if arm != 'baseline':
                entry[arm]['paired_vs_baseline'] = {
                    'overall': paired(outcomes['baseline'], outcome),
                    'mirrored_signers': paired(outcomes['baseline'], outcome, mirrored),
                    'unmirrored_signers': paired(outcomes['baseline'], outcome, unmirrored)}
        results[method] = entry
    arm = 'armA_canonical_refit'
    h1 = {m: results[m][arm]['mirrored_signers']['top1'] > results[m]['baseline']['mirrored_signers']['top1']
          for m in METHODS}
    h2 = {m: results[m][arm]['overall']['top1'] >= results[m]['baseline']['overall']['top1'] for m in METHODS}
    svm_overall = results['linear_svm'][arm]['overall']['top1'] > results['linear_svm']['baseline']['overall']['top1']
    verdict = ('supported' if all(h1.values()) and svm_overall
               else 'partially_supported' if any(h1.values()) else 'not_supported')
    analysis = {'schema_version': 1, 'status': 'verified', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'test_used': False, 'validation_rows': len(rows), 'mirrored_validation_rows': sum(mirrored),
                'mirrored_validation_signers': sorted({r['signer_id'] for r, m in zip(rows, mirrored) if m}),
                'mirrored_train_signers': sorted(k for k, v in decisions['train'].items() if v['mirror']),
                'decisions': decisions, 'results': results, 'checks': checks,
                'preregistered_evaluation': {'H1_mirrored_subset_improves': h1, 'H2_overall_not_reduced': h2,
                                             'linear_svm_overall_improves': svm_overall, 'verdict': verdict},
                'caveats': ['Hypothesis generated post hoc from the same validation per-signer table; not an independent confirmation.',
                            'Signer-level decision uses signer IDs and unlabeled clips of each signer (declared, label-free).',
                            'McNemar treats clips as independent although they cluster within six signers.',
                            'kNN Top-5 counts only voted classes and is not comparable with SVM/RF Top-5.'],
                'inputs': inputs}
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + '\n')
    lines = ['| Method | Arm | Overall Top-1 | Mirrored signers Top-1 | Unmirrored Top-1 | Net vs baseline (overall / mirrored) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        for name, entry in results[method].items():
            pair = entry.get('paired_vs_baseline')
            net = f"{pair['overall']['net']:+d} / {pair['mirrored_signers']['net']:+d}" if pair else '—'
            lines.append(f"| {method} | {name} | {100 * entry['overall']['top1']:.4f}% | "
                         f"{100 * entry['mirrored_signers']['top1']:.4f}% | {100 * entry['unmirrored_signers']['top1']:.4f}% | {net} |")
    lines.append(f"\nVerdict (preregistered rule): **{verdict}**")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(analysis['preregistered_evaluation']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, required=True)
    parser.add_argument('--arm0-dir', type=Path, required=True)
    parser.add_argument('--arma-experiment', type=Path, required=True)
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-experiment', type=Path,
                        default=WORKSPACE / 'runs/20261002T161254Z-classical-formal-three-methods-dac0c02f/experiment')
    run(parser.parse_args())
