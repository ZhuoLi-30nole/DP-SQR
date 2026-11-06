"""Check archive integrity, manuscript table values, and unchanged algorithms."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from run import ROOT
import numpy as np
import pandas as pd
from dpsqr.study import source_digest
from make_tables import build


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_rows(path):
    # Fixtures are never executed or used to generate numerical outputs.
    text = path.read_text().split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
    return [line.strip().removesuffix(r'\\').split('&')
            for line in text.splitlines() if '&' in line]


def verify(output):
    archive = json.loads((ROOT / 'paper_results/manifest.json').read_text())
    assert source_digest() == archive.get('release_source_digest', archive['source_digest']), 'Algorithm/runner source changed'
    for entry in archive['figure_sources']:
        assert sha(ROOT / entry['release']) == entry['sha256'], entry['release']
    for entry in archive['figures']:
        assert sha(ROOT / 'reference/figures' / entry['paper_file']) == entry['paper_sha256']
    if (ROOT / 'RELEASE_MANIFEST.json').exists():
        release = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
        for entry in release['files']:
            assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
    build(ROOT / 'paper_results', output)
    ref = ROOT / 'reference/tables'
    for name in ['low_prediction_error', 'low_prediction_with_frappe']:
        frame = pd.read_csv(output / f'{name}.csv')
        expected = [re.findall(r'\\mbox\{([^}]+)\}', '&'.join(row))
                    for row in table_rows(ref / f'{name}.tex')]
        assert frame.iloc[:, 2:].values.tolist() == expected, name
    for name in ['housing', 'housing_with_frappe']:
        frame = pd.read_csv(output / f'{name}.csv', dtype=str)
        expected = [[x.strip() for x in row[1:]]
                    for row in table_rows(ref / 'real_coefficients' / f'{name}.tex')]
        assert frame.iloc[:, 1:].values.tolist() == expected, name
    frame = pd.read_csv(output / 'coverage_combined.csv', dtype=str)
    expected = [[x.strip() for x in row[2:]]
                for row in table_rows(ref / 'tab_coverage-combined.tex')]
    assert frame.iloc[:, 2:].values.tolist() == expected, 'coverage'
    frame = pd.read_csv(output / 'high_excess_loss.csv', dtype=str)
    expected = [[x.strip() for x in row[1:]]
                for row in table_rows(ref / 'tab_sim-high-pe.tex')]
    assert frame.iloc[:, 1:].values.tolist() == expected, 'high excess loss'
    for name, fixture in [('screening_main', 'tab_screening-stress.tex'),
                          ('screening_supp', 'tab_supp-screening-stress.tex')]:
        frame = pd.read_csv(output / f'{name}.csv', dtype=str)
        expected = [[x.strip() for x in row[2:]] for row in table_rows(ref / fixture)]
        assert frame.iloc[:, 2:].values.tolist() == expected, name
    paired = pd.read_csv(output / 'paired_real.csv')
    expected = [[x.strip().strip('$') for x in row[-2:]]
                for row in table_rows(ref / 'tab_real-data-paired.tex')]
    # Match the displayed rounding precision, not unavailable extra digits.
    for actual, target in zip(paired[['mean', 'se']].values, expected, strict=True):
        for value, displayed in zip(actual, target, strict=True):
            if 'e' in displayed.lower():
                mantissa = displayed.lower().split('e')[0]
                digits = len(mantissa.replace('.', '').lstrip('-').lstrip('0'))
                rounded = float(f'{value:.{digits}g}')
            else:
                decimals = len(displayed.split('.')[1]) if '.' in displayed else 0
                rounded = float(f'{value:.{decimals}f}')
            assert rounded == float(displayed), (value, displayed)
    print('PASS: source digest, archive hashes, 12 figure fixtures, and 9 numeric tables')
    return dict(status='passed', source_digest=source_digest(),
                figure_fixtures=12, numeric_tables=9,
                scope='Archived-data reconstruction; full 300-repeat fits were not rerun')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/tables')
    args = p.parse_args()
    report = verify(args.output)
    (args.output.parent / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
