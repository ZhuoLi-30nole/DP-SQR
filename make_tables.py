"""Recompute manuscript table values from archived individual fits.

LaTeX outputs use a simple portable layout; reference/tables preserves the
manuscript layouts separately. No reference table supplies computed values.
"""
import argparse
import json
from pathlib import Path

from run import ROOT
import numpy as np
import pandas as pd
from dpsqr.display_names import publication_frame

LOW = ['sqr_gd', 'sqr_gd_np', 'tran_dpsgd', 'tran_opm', 'huber_gd']
HIGH = ['sqr_sr', 'sqr_sr_np', 'sqr_iht', 'sqr_iht_np', 'huber_iht', 'cai_iht']
SCREEN = ['sqr_sr_np', 'sqr_sr', 'sqr_iht_np', 'sqr_iht', 'huber_iht']
SCENARIOS = [('reference_ar', .1), ('strong_ar', .8), ('strong_ar', .95),
             ('cancellation', .8), ('cancellation', .95),
             ('suppressor', .8), ('suppressor', .95)]
FEATURES = ['Intercept', 'median_income', 'housing_median_age',
            'population', 'households', 'total_rooms']


def read_plot(source, name):
    frame = pd.read_csv(source / 'plot_sources' / f'{name}_source.csv.gz')
    if frame.task_id.duplicated().any() or not frame.status.eq('ok').all():
        raise ValueError(f'Invalid archived fits: {name}')
    return frame


def select(frame, method, epsilon, **kwargs):
    part = frame.loc[frame.method.eq(method)]
    if not method.endswith('_np'):
        part = part.loc[np.isclose(part.epsilon, epsilon)]
    for key, value in kwargs.items():
        part = part.loc[part[key].eq(value)]
    if len(part) != 300 or part.rep.nunique() != 300:
        raise ValueError(f'Expected 300 distinct paired repetitions: {method}, {kwargs}')
    return part.sort_values('rep')


def mean_se(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 300 or not np.isfinite(values).all():
        raise ValueError('A manuscript cell requires 300 finite measurements')
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def publish(frame, output, name):
    frame.to_csv(output / f'{name}.csv', index=False)
    publication_frame(frame).to_latex(output / f'{name}.tex', index=False, escape=True)


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    low = read_plot(source, 'sim_low_overview')
    for name, methods in [('low_prediction_error', LOW),
                          ('low_prediction_with_frappe', LOW + ['frappe'])]:
        rows = []
        for noise in ['gaussian', 't3', 'cauchy']:
            for epsilon in [.1, .5, .9]:
                row = dict(noise=noise, epsilon=epsilon)
                for method in methods:
                    mean, se = mean_se(select(low, method, epsilon, n_train=1000,
                                              tau=.5, noise=noise).excess_ql)
                    row[method] = f'{mean:.3f} ({se:.2g})'
                rows.append(row)
        publish(pd.DataFrame(rows), output, name)

    coefficients = pd.read_csv(source / 'housing_coefficients.csv.gz')
    for name, methods in [('housing', LOW + ['lad', 'ols']),
                          ('housing_with_frappe', LOW + ['frappe', 'lad', 'ols'])]:
        rows = []
        for feature in FEATURES:
            row = dict(feature=feature)
            for method in methods:
                part = coefficients.loc[coefficients.method.eq(method)
                                        & coefficients.feature.eq(feature)]
                mean, _ = mean_se(part.coefficient)
                row[method] = f'{mean:.5f}'
            rows.append(row)
        publish(pd.DataFrame(rows), output, name)

    high = read_plot(source, 'sim_high_core')
    rows = []
    for n in [5000, 7500, 10000]:
        for epsilon in [.5, .75, 1.]:
            part = select(high, 'sqr_sr', epsilon, n_train=n)
            rows.append(dict(n_train=n, epsilon=epsilon,
                             recall=f'{part.recall.mean():.3f}',
                             full_coverage=f'{int(part.support_coverage.sum())}/300'))
    publish(pd.DataFrame(rows), output, 'coverage_combined')
    rows = []
    for method in HIGH:
        row = dict(method=method)
        for epsilon in [.5, .75, 1.]:
            mean, se = mean_se(select(high, method, epsilon, n_train=10000).excess_ql)
            row[f'epsilon_{epsilon:g}'] = f'{mean:.4f} ({se:.2g})'
        rows.append(row)
    publish(pd.DataFrame(rows), output, 'high_excess_loss')

    screening = pd.read_csv(source / 'screening_stress.csv.gz')
    rows = []
    for scenario, rho in SCENARIOS:
        row = dict(scenario=scenario, rho=rho)
        for method in SCREEN:
            part = screening.loc[screening.scenario.eq(scenario)
                                 & screening.rho.eq(rho) & screening.method.eq(method)]
            if part.rep.nunique() != 300:
                raise ValueError('Incomplete screening repetitions')
            mean, se = mean_se(part.recall)
            row[method] = f'{mean:.3f} ({se:.3f})'
        rows.append(row)
    screen = pd.DataFrame(rows)
    publish(screen, output, 'screening_supp')
    publish(screen.drop(columns='sqr_iht_np'), output, 'screening_main')

    rows = []
    comparisons = [('housing', 8000, .3, 'sqr_gd', 'tran_opm'),
                   ('housing', 16000, .9, 'sqr_gd', 'tran_opm'),
                   ('housing', 16000, .9, 'sqr_gd', 'sqr_gd_np'),
                   ('crime', 1520, .9, 'sqr_sr', 'sqr_sr_np'),
                   ('crime', 1520, .9, 'sqr_iht', 'sqr_iht_np')]
    for dataset, n, epsilon, first, reference in comparisons:
        frame = read_plot(source, f'real_{dataset}_quantile_grid')
        a = select(frame, first, epsilon, n_train=n, tau=.5).set_index('rep')
        b = select(frame, reference, epsilon, n_train=n, tau=.5).set_index('rep')
        if not a.index.equals(b.index):
            raise ValueError('Unpaired training splits')
        mean, se = mean_se(a.ql_tau - b.ql_tau)
        rows.append(dict(dataset=dataset, n_train=n, epsilon=epsilon,
                         first=first, reference=reference, mean=mean, se=se))
    publish(pd.DataFrame(rows), output, 'paired_real')
    design = pd.DataFrame([
        dict(dataset='housing', available_records=20640, p=6,
             n_train='4000, 8000, 12000, 16000',
             n_test='1000, 2000, 3000, 4000'),
        dict(dataset='crime', available_records=1994, p=99,
             n_train='400, 640, 880, 1200, 1520',
             n_test='100, 160, 220, 300, 380')])
    for row in design.to_dict('records'):
        c = json.loads((ROOT / 'configs' / f"real_{row['dataset']}.json").read_text())
        assert row['n_train'] == ', '.join(str(int(m * c['train_fraction'])) for m in c['sample_sizes'])
        assert row['n_test'] == ', '.join(str(m - int(m * c['train_fraction'])) for m in c['sample_sizes'])
    publish(design, output, 'real_design')
    print(f'Recomputed 10 manuscript tables in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'paper_results')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/tables')
    args = parser.parse_args()
    build(args.source, args.output)
