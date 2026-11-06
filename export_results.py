"""Project newly completed run outputs into compact plotting/table inputs."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from run import ROOT

METRIC_COLUMNS = {
    'sample_m', 'n', 'tau', 'rep', 'study', 'regime', 'method', 'epsilon',
    'delta', 'n_train', 'n_test', 'p', 'seed', 'data_key', 'task_id', 'role',
    'status', 'protocol', 'diagnostic_only', 'ql_tau', 'response_units',
    'true_sparsity', 'noise', 'l2_error', 'excess_ql', 'recall',
    'support_coverage', 'selected_slopes',
}


def project(src, dst, columns=None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(newline='') as inp, gzip.open(dst, 'wt', newline='') as out:
        reader = csv.DictReader(inp)
        fields = [x for x in reader.fieldnames if columns is None or x in columns]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            writer.writerow({x: row[x] for x in fields})


def export(processed, screening, output):
    if output.resolve() == (ROOT / 'paper_results').resolve():
        raise ValueError('Keep the shipped paper_results archive immutable; choose a new destination')
    if not (processed / 'figure_manifest.json').exists():
        raise ValueError('First run run.py figures on the completed formal runs')
    manifest = json.loads((processed / 'figure_manifest.json').read_text())
    if manifest['diagnostic']:
        raise ValueError('Formal manuscript reproduction requires completed formal inputs')
    config = json.loads((screening / 'config.json').read_text())
    required = dict(n=5000, p=6000, test_n=5000, repeats=300, epsilon=1.,
                    size=12, seed=2026110601, noise='t3')
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Screening settings differ from the manuscript')
    output.mkdir(parents=True, exist_ok=True)
    for path in processed.glob('*_source.csv'):
        project(path, output / 'plot_sources' / (path.name + '.gz'), METRIC_COLUMNS)
    project(processed / 'real_housing_coefficients_raw.csv',
            output / 'housing_coefficients.csv.gz')
    fields = set(next(csv.reader((screening / 'raw.csv').open())))
    fields -= {'parameters', 'privacy', 'metadata', 'seconds'}
    project(screening / 'raw.csv', output / 'screening_stress.csv.gz', fields)
    project(screening / 'rank_diagnostics.csv', output / 'screening_rank_diagnostics.csv.gz')
    print(f'Exported new compact inputs to {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed', type=Path, required=True)
    p.add_argument('--screening', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/recomputed_results')
    args = p.parse_args()
    export(args.processed, args.screening, args.output)
