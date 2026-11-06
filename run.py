"""Single command-line/PyCharm entry point. See README.md."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.runtime/cache'))
import argparse
import json
import sys
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description='DP-SQR fixed-rule simulations and public-data experiments')
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['plan','run']:
        q=sub.add_parser(command);q.add_argument('--config',required=True)
        q.add_argument('--methods',nargs='+',help='Explicit subset, recorded as a different protocol config')
        q.add_argument('--repeats',type=int)
        if command=='run':
            q.add_argument('--output',required=True);q.add_argument('--workers',type=int,default=1)
            q.add_argument('--max-jobs',type=int,help='Diagnostic prefix only; never a complete formal grid')
    q=sub.add_parser('figures');q.add_argument('--input',nargs='+',required=True)
    q.add_argument('--output',required=True);q.add_argument('--allow-partial',action='store_true')
    q=sub.add_parser('check');q.add_argument('--tests',action='store_true')
    q=sub.add_parser('batch');q.add_argument('--study',choices=['low','high','real','all','smoke'],default='all')
    q.add_argument('--output-root',default='results/formal');q.add_argument('--workers',type=int,default=1)
    q.add_argument('--repeats',type=int)
    a=p.parse_args()
    if a.command in ('plan','run'):
        from dpsqr.study import plan,run
        cfg=Path(a.config)
        if not cfg.exists():cfg=ROOT/'configs'/(a.config.removesuffix('.json')+'.json')
        c=json.loads(cfg.read_text())
        if a.methods:c['methods']=a.methods
        if a.repeats is not None:c['repeats']=a.repeats
        if a.command=='plan': print(json.dumps(plan(c),indent=2,ensure_ascii=False))
        else:
            if a.max_jobs is not None and a.max_jobs<1:p.error('--max-jobs must be positive')
            run(c,a.output,a.workers,a.max_jobs)
    elif a.command=='batch':
        from dpsqr.study import run
        groups={'low':['sim_low_p5','sim_low_p10','sim_low_p20'],
                'high':['sim_high_core','sim_high_gaussian_quantiles','sim_high_tails','sim_high_support'],
                'real':['real_housing','real_crime'],
                'smoke':['smoke_sim_low','smoke_sim_high','smoke_real_housing','smoke_real_crime']}
        names=groups['low']+groups['high']+groups['real'] if a.study=='all' else groups[a.study]
        for name in names:
            c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if a.repeats is not None:c['repeats']=a.repeats
            run(c,Path(a.output_root)/name,a.workers)
    elif a.command=='figures':
        from dpsqr.reporting import build
        build(a.input,a.output,allow_partial=a.allow_partial)
    else:
        from dpsqr.study import plan,benchmark
        for cfg in sorted((ROOT/'configs').glob('*.json')):
            plan(json.loads(cfg.read_text()))
        for ds in ['housing','crime']:
            X,y,_=benchmark(ds);print(ds,X.shape,len(y))
        if a.tests:
            import unittest
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
            if not result.wasSuccessful():raise SystemExit(1)
        print('Configuration/data checks passed')


if __name__=='__main__':main()
