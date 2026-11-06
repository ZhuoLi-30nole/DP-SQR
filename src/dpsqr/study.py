"""Reproducible paired experiments and atomic per-fit resume checkpoints.

This runner is for synthetic/public benchmark research. Test losses, seeds,
splits and coefficients across repetitions are NOT a joint private release.
"""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import itertools
import json
import math
import os
import platform
import time
import traceback
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from .policy import VERSION, LOW, HIGH, total_delta, parameters, methods_for, allocation
from .pipeline import fit_pipeline
from .simulation import generate
from .core import check_loss
from .benchmark_data import load_housing, load_crime

ROOT=Path(__file__).resolve().parents[2]


def json_value(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=json_value).encode()).hexdigest()


def derived_seed(seed,*parts):
    return int(digest([seed,*parts])[:8],16)


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=json_value)+'\n')
    os.replace(tmp,path)


def source_digest():
    files=sorted((ROOT/'src').rglob('*.py'))+[ROOT/'run.py']
    return digest([(str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])


def data_digest():
    return digest([(str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted((ROOT/'data').rglob('*')) if p.is_file()])


def validate_config(c):
    if c.get('version')!=VERSION or c.get('hyperparameter_search') is not False:
        raise ValueError('Use a current fixed-rules config; search is not part of this protocol')
    if c.get('delta_rule')!='10*n**(-1.1)': raise ValueError('Unexpected delta rule')
    if c.get('initializer_projection','missing') is not None: raise ValueError('Ridge initialization has no radius projection')
    if type(c.get('repeats')) is not int or c['repeats']<1: raise ValueError('Invalid repeats')
    if any(not 0<e<=1 for e in c['epsilon']): raise ValueError('epsilon must lie in (0,1]')
    if any(t not in [.25,.5,.75] for t in c['tau']): raise ValueError('Unexpected quantile')
    if c['study']=='real':
        if c['regime'] not in ('housing','crime') or c['train_fraction']!=.8:
            raise ValueError('Real studies require the two benchmarks and an 80/20 split')
        if c['preprocessing']!='training_only_public_benchmark': raise ValueError('Unexpected preprocessing scope')
        if c['regime']=='crime' and c['working_sparsity']!=5: raise ValueError('Real sparse dimension includes intercept and equals 5')
        limit=20640 if c['regime']=='housing' else 1994
        if any(type(m) is not int or m<20 or m>limit for m in c['sample_sizes']): raise ValueError('Invalid sample size')
    elif c['study']=='simulation':
        d=c['design']
        if any(type(n) is not int or n<20 for n in d['n']): raise ValueError('Invalid n')
        if c['regime']=='low' and (len(d['p'])!=1 or d['p']!=d['sparsity']): raise ValueError('Low-dimensional design must be dense')
        if c['regime']=='high':
            if max(d['sparsity'])>c['working_sparsity'] or c['working_sparsity']>min(d['p']): raise ValueError('Invalid public working dimension')
    else: raise ValueError('Unknown study')
    known=set(LOW+HIGH)
    if c.get('methods') and not set(c['methods'])<=known: raise ValueError('Unknown method filter')


def tasks(c):
    if c['study']=='simulation':
        d=c['design']
        for n,p,s,tau,noise,rep in itertools.product(d['n'],d['p'],d['sparsity'],c['tau'],d['noise'],range(c['repeats'])):
            yield dict(n=n,p=p,true_sparsity=s,tau=tau,noise=noise,rep=rep)
    else:
        for m,tau,rep in itertools.product(c['sample_sizes'],c['tau'],range(c['repeats'])):
            yield dict(sample_m=m,n=int(.8*m),tau=tau,rep=rep)


def task_methods(c,t):
    names=methods_for(c['regime'],t['tau'],t.get('noise'))
    return [m for m in names if not c.get('methods') or m in c['methods']]


def coefficient_target(c,t):
    return c['study']=='real' and c['regime']=='housing' and t['sample_m']==c['coefficient_sample_size'] and t['tau']==c['coefficient_tau']


def is_diagnostic(c):
    return bool(c.get('diagnostic_only',False) or c['repeats']!=300 or c.get('methods'))


def plan(c):
    validate_config(c)
    jobs=fits=0;counts={}
    for t in tasks(c):
        jobs+=1
        for m in task_methods(c,t):
            k=1 if m.endswith('_np') else len(c['epsilon'])
            counts[m]=counts.get(m,0)+k;fits+=k
        if coefficient_target(c,t):
            for m in ['lad','ols']: counts[m]=counts.get(m,0)+1;fits+=1
    return dict(scenario_jobs=jobs,complete_fits=fits,method_fit_counts=counts,
                repeats=c['repeats'],diagnostic_only=is_diagnostic(c),
                note='nonDP fitted once per shared sample, not once per epsilon; LAD/OLS coefficient-only')


@lru_cache(maxsize=2)
def benchmark(regime):
    return (load_housing if regime=='housing' else load_crime)(ROOT/'data')


def prepare_real(c,t):
    X,y,meta=benchmark(c['regime'])
    seed=derived_seed(c['seed'],'records-and-split',c['regime'],t['sample_m'],t['rep'])
    # One random permutation supplies a random m-record sample AND its 80/20 split.
    ids=np.random.default_rng(seed).permutation(len(y))[:t['sample_m']]
    tr,te=ids[:t['n']],ids[t['n']:]
    xmean=X[tr].mean(axis=0);xsd=X[tr].std(axis=0);xsd=np.where(xsd>0,xsd,1.)
    ym=float(y[tr].mean())
    ys=float(y[tr].std()) if c['regime']=='crime' else 1.
    if ys<=0: ys=1.
    train=np.column_stack((np.ones(len(tr)),(X[tr]-xmean)/xsd))
    test=np.column_stack((np.ones(len(te)),(X[te]-xmean)/xsd))
    transform=dict(x_mean=xmean,x_sd=xsd,y_mean=ym,y_sd=ys)
    split_info=dict(seed=seed,training_ids_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
                    test_ids_sha256=hashlib.sha256(te.tobytes()).hexdigest(),
                    split_disjoint=not bool(np.intersect1d(tr,te).size),n_train=len(tr),n_test=len(te),
                    scaling_scope='training-only public benchmark; not raw-record DP preprocessing',
                    **transform)
    return train,(y[tr]-ym)/ys,test,(y[te]-ym)/ys,transform,meta,split_info


def original_coefficients(b,transform):
    slopes=b[1:]*transform['y_sd']/transform['x_sd']
    intercept=transform['y_mean']+transform['y_sd']*b[0]-transform['x_mean']@slopes
    return np.r_[intercept,slopes]


def lad_fit(X,y):
    """Exact unpenalized LAD by a sparse linear program, including intercept."""
    n,p=X.shape
    A=sparse.vstack([sparse.hstack([sparse.csr_matrix(X),-sparse.eye(n)]),
                     sparse.hstack([-sparse.csr_matrix(X),-sparse.eye(n)])],format='csr')
    result=linprog(np.r_[np.zeros(p),np.ones(n)/n],A_ub=A,b_ub=np.r_[y,-y],
                   bounds=[(None,None)]*p+[(0,None)]*n,method='highs')
    if not result.success: raise RuntimeError('LAD solver: '+result.message)
    return result.x[:p]


def budget_check(fit,epsilon,delta):
    ledger=fit.privacy
    if ledger.get('type')=='nonprivate': return
    # FRAPPE amplified/global stages, not the unamplified internal budget.
    stages=ledger.get('amplified_stages') or ledger.get('stages',[])
    if stages:
        e=sum(float(s['epsilon']) for s in stages)
        d=sum(float(s.get('delta',0)) for s in stages)
        if e>epsilon+1e-10 or d>delta+1e-12:
            raise RuntimeError(f'Stage composition exceeds total budget: {e},{d}')


def run_task(payload):
    c,t,out=payload
    out=Path(out)
    # Runtime limits apply to Python BLAS and scipy/HiGHS where supported.
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError: pass
    real=c['study']=='real'
    if real:
        X,y,Xt,yt,transform,meta,split=prepare_real(c,t)
        atomic_json(out/'splits'/f"{c['regime']}_{t['sample_m']}_{t['rep']}.json",split)
        data_key=digest([split['training_ids_sha256'],split['test_ids_sha256']])
        p=X.shape[1]
    else:
        high=c['regime']=='high'
        kw=dict(p=t['p'],sparsity=t['true_sparsity'],tau=t['tau'],noise=t['noise'],
                signal=1. if high else .6,correlation=.1 if high else 0.,intercept=high,
                design_distribution='gaussian' if high else 'uniform_copula',
                coefficient_pattern='fixed_rademacher' if high else 'alternating',coefficient_seed=0)
        # Exclude epsilon and tau: identical X and unshifted errors across both.
        seed=derived_seed(c['seed'],'data',c['regime'],t['n'],t['p'],t['true_sparsity'],t['noise'],t['rep'])
        sample=generate(t['n'],seed=seed,**kw)
        test=generate(c['test_n'],seed=derived_seed(seed,'test'),**kw)
        X,y,Xt,yt=sample.X,sample.y,test.X,test.y;p=t['p'];data_key=str(seed)
    names=task_methods(c,t)
    if coefficient_target(c,t): names+=['lad','ols']
    delta=total_delta(t['n']);rows=[]
    for name in names:
        epsilons=[None] if name.endswith('_np') or name in ('lad','ols') else c['epsilon']
        for eps in epsilons:
            key=digest([t,name,eps])[:24];path=out/'checkpoints'/(key+'.json')
            if path.exists():
                rows.append(json.loads(path.read_text()));continue
            seed=derived_seed(c['seed'],'algorithm',c['regime'],t,name.removesuffix('_np'),eps)
            row=dict(t,study=c['study'],regime=c['regime'],method=name,epsilon=eps,
                     delta=delta,n_train=t['n'],n_test=len(yt),p=p,seed=seed,data_key=data_key,
                     task_id=key,role='coefficient_only' if name in ('lad','ols') else 'prediction',
                     status='failed',protocol=VERSION,diagnostic_only=is_diagnostic(c),
                     privacy_scope='per-fit processed-input accounting; not joint release or private evaluation',
                     parameter_source='fixed public rules and budgeted scale; no candidate search')
            start=time.perf_counter()
            try:
                if name in ('lad','ols'):
                    b=lad_fit(X,y) if name=='lad' else np.linalg.lstsq(X,y,rcond=None)[0]
                    row.update(parameters='{}',privacy={'type':'nonprivate'},metadata={})
                    support=np.flatnonzero(b)
                else:
                    par=parameters(name,n=t['n'],p=p,regime=c['regime'],s=c['working_sparsity'],intercept=real or c['regime']=='high')
                    row['parameters']=par
                    fit=fit_pipeline(name,X,y,par,epsilon=eps if eps is not None else c['epsilon'][0],
                                     delta=delta,tau=t['tau'],seed=seed)
                    if fit.metadata.get('optimizer_success') is False: raise RuntimeError('Numerical optimizer did not converge')
                    b=np.asarray(fit.coef);support=np.asarray(fit.support)
                    budget_check(fit,eps if eps is not None else c['epsilon'][0],delta)
                    row.update(privacy=fit.privacy,metadata=fit.metadata)
                if b.shape!=(p,) or not np.isfinite(b).all(): raise RuntimeError('Invalid coefficients')
                if real:
                    # Report loss in the original response units (log-price or crime rate).
                    row['ql_tau']=float(np.mean(check_loss(yt-Xt@b,t['tau']))*transform['y_sd'])
                    row['response_units']=meta['response']
                    if coefficient_target(c,t) and (eps==c['coefficient_epsilon'] or eps is None):
                        row['coefficients']=original_coefficients(b,transform).tolist()
                        row['features']=['Intercept',*meta['predictors']]
                else:
                    err=b-sample.beta
                    # Stable per-observation difference avoids subtracting two huge Cauchy losses.
                    residual=yt-Xt@sample.beta;shift=Xt@err
                    excess=t['tau']*(-shift)+np.maximum(-residual+shift,0)-np.maximum(-residual,0)
                    row.update(l2_error=float(np.linalg.norm(err)),excess_ql=float(np.mean(excess)))
                    if c['regime']=='high':
                        truth=set(np.flatnonzero(sample.beta));selected=set(map(int,support))
                        truth.discard(0);selected.discard(0)
                        row.update(recall=len(truth&selected)/len(truth),support_coverage=float(truth<=selected),
                                   selected_slopes=len(selected))
                row['status']='ok'
            except Exception as exc:
                row.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            row['seconds']=time.perf_counter()-start
            atomic_json(path,row);rows.append(row)
    return rows


def summarize(rows,out):
    frame=pd.DataFrame(rows);out=Path(out)
    flat=frame.copy()
    for col in ['parameters','privacy','metadata','coefficients','features']:
        if col in flat:
            flat[col]=flat[col].map(lambda v: json.dumps(v,default=json_value) if isinstance(v,(dict,list)) else v)
    flat.to_csv(out/'raw.csv',index=False)
    group=[k for k in ['study','regime','n_train','sample_m','p','true_sparsity','noise','tau','epsilon','method','role'] if k in frame]
    counts=frame.groupby(group,dropna=False).status.agg(n_attempted='size',n_ok=lambda s:(s=='ok').sum()).reset_index()
    counts['n_failed']=counts.n_attempted-counts.n_ok
    for metric in ['l2_error','excess_ql','recall','support_coverage','ql_tau']:
        if metric not in frame:continue
        good=frame[(frame.status=='ok')&frame[metric].notna()]
        vals=good.groupby(group,dropna=False)[metric].agg(['mean','std','count']).reset_index()
        vals[metric+'_mcse']=vals['std']/np.sqrt(vals['count'])
        vals=vals.rename(columns={'mean':metric+'_mean','std':metric+'_sd','count':metric+'_n'})
        counts=counts.merge(vals,on=group,how='left')
    counts.to_csv(out/'summary.csv',index=False)
    coefficients=[]
    for row in rows:
        if row['status']=='ok' and 'coefficients' in row:
            for feature,value in zip(row['features'],row['coefficients']):
                coefficients.append({k:row[k] for k in ['rep','method','epsilon','n_train','tau']}|dict(feature=feature,coefficient=value))
    if coefficients:
        f=pd.DataFrame(coefficients);f.to_csv(out/'coefficients_raw.csv',index=False)
        tab=f.groupby(['n_train','tau','epsilon','method','feature'],dropna=False).coefficient.agg(['mean','std','count']).reset_index()
        tab.to_csv(out/'coefficients_summary.csv',index=False)
        wide=tab.assign(estimate=tab.apply(lambda r:f"{r['mean']:.4f} ({r['std']:.4f})",axis=1)).pivot(index='method',columns='feature',values='estimate')
        wide=wide.reindex(columns=['Intercept',*[x for x in wide.columns if x!='Intercept']])
        wide.to_latex(out/'coefficients_table.tex',escape=True,caption='Median coefficients: mean (SD), epsilon=0.9 for DP methods.')
    return frame


def run(c,out,workers=1,max_jobs=None):
    info=plan(c);out=Path(out).resolve()
    identity=dict(config=c,source_digest=source_digest(),data_digest=data_digest() if c['study']=='real' else None)
    fingerprint=digest(identity)
    marker=out/'run_spec.json'
    if out.exists() and not marker.exists() and any(out.iterdir()): raise ValueError('Nonempty output directory has no run specification')
    if marker.exists() and json.loads(marker.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Code/config/data changed. Use a NEW output directory; old results cannot be mixed.')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(marker,dict(fingerprint=fingerprint,**identity))
    todo=list(itertools.islice(tasks(c),max_jobs)) if max_jobs else list(tasks(c))
    # Conservative cap: data, test arrays and several gradient buffers per worker.
    if c['study']=='simulation':
        bytes_per=8*max(c['design']['p'])*(6*max(c['design']['n'])+2*c['test_n'])
    else:bytes_per=128*1024**2
    try:
        import psutil
        memory_workers=max(1,int(.6*psutil.virtual_memory().available/max(bytes_per,1)))
    except ImportError:memory_workers=1
    workers=max(1,min(workers,memory_workers,len(todo)))
    atomic_json(out/'manifest.json',dict(protocol=VERSION,plan=info,selected_jobs=len(todo),workers=workers,
                estimated_bytes_per_worker=bytes_per,python=platform.python_version(),
                benchmark_preprocessing='training-only; public-data research, not raw-record DP preprocessing',
                complete=False,full_grid=max_jobs is None))
    print(f"{len(todo)} scenario jobs; {info['complete_fits']} fits in full config; workers={workers}",flush=True)
    rows=[]
    if workers==1:
        for i,t in enumerate(todo):
            rows.extend(run_task((c,t,str(out))))
            print(f"completed scenario {i+1}/{len(todo)}",flush=True)
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
            iterator=iter(todo);pending=set()
            for _ in range(min(workers*2,len(todo))):pending.add(pool.submit(run_task,(c,next(iterator),str(out))))
            done_count=0
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for job in done:
                    rows.extend(job.result());done_count+=1
                    print(f"completed scenario {done_count}/{len(todo)}",flush=True)
                    t=next(iterator,None)
                    if t is not None:pending.add(pool.submit(run_task,(c,t,str(out))))
    rows.sort(key=lambda r:r['task_id'])
    frame=summarize(rows,out)
    manifest=json.loads((out/'manifest.json').read_text())
    manifest.update(complete=True,n_rows=len(rows),n_failed=int((frame.status!='ok').sum()))
    atomic_json(out/'manifest.json',manifest)
    print(f"Saved {out}; failures={manifest['n_failed']}",flush=True)
    return frame
