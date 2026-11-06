"""Rebuild the archived paper outputs, or rerun all manuscript experiments."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = ['sim_low_p5', 'sim_low_p10', 'sim_low_p20', 'sim_high_core',
          'sim_high_gaussian_quantiles', 'sim_high_tails', 'sim_high_support',
          'real_housing', 'real_crime']


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['archived', 'full'], default='archived')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--results-root', type=Path, default=ROOT / 'results/formal')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    out = args.output.resolve()
    source = ROOT / 'paper_results'
    if args.mode == 'full':
        result_root = args.results_root.resolve()
        run('run.py', 'batch', '--study', 'all', '--output-root', result_root,
            '--workers', args.workers)
        screening = result_root.parent / 'screening_stress_t3_r300_test5000'
        run('scripts/screening_stress.py', '--noise', 't3', '--test-n', 5000,
            '--repeats', 300, '--workers', args.workers, '--output', screening)
        processed = out / 'processed'
        run('run.py', 'figures', '--input', *[result_root / n for n in GROUPS],
            '--output', processed)
        source = out / 'recomputed_results'
        run('export_results.py', '--processed', processed, '--screening', screening,
            '--output', source)
    run('make_figures.py', '--source', source / 'plot_sources',
        '--output', out / 'figures')
    run('make_tables.py', '--source', source, '--output', out / 'tables')
    if args.mode == 'archived':
        run('verify_release.py', '--output', out / 'tables')


if __name__ == '__main__':
    main()
