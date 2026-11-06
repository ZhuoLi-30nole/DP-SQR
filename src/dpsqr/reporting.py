"""Figures follow the revised experiment contract; no archived values are read.

Quantitative grids, method-consistent colors, mean +/- one MCSE for curves,
median/IQR/1.5-IQR whiskers and ALL fliers for boxplots. Export editable PDF/SVG
and 300-dpi PNG previews, plus the actual per-fit source rows for every figure.
"""
from pathlib import Path
import json
import string
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .policy import LOW,HIGH
from .study import atomic_json

LABELS={'sqr_gd':'DP-SQR-GD','sqr_gd_np':'nonDP-SQR-GD','sqr_sr':'DP-SQR-SR',
        'sqr_sr_np':'nonDP-SQR-SR','sqr_iht':'DP-SQR-IHT','sqr_iht_np':'nonDP-SQR-IHT',
        'tran_dpsgd':'Tran-SGD','tran_opm':'Tran-OPM','huber_gd':'DP-Huber-GD',
        'huber_iht':'DP-Huber-IHT','cai_iht':'Cai-IHT','frappe':'FRAPPE'}
COLORS={'sqr_gd':'#377eb8','sqr_gd_np':'#777777','sqr_sr':'#377eb8','sqr_sr_np':'#777777',
        'sqr_iht':'#e69f00','sqr_iht_np':'#bdbdbd','tran_dpsgd':'#cc79a7','tran_opm':'#009e73',
        'huber_gd':'#8e6bb0','huber_iht':'#8e6bb0','cai_iht':'#56b4e9','frappe':'#a6761d'}
NOISES=['gaussian','t3','cauchy']
NOISE_LABELS={'gaussian':'Gaussian','t3':r'Student $t_3$','t2.25':r'Student $t_{2.25}$','cauchy':'Cauchy'}


def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':6.5,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.right':False,
                         'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False})


def select(f,**conditions):
    out=f
    for k,v in conditions.items():
        if k not in out:return out.iloc[0:0]
        if k=='epsilon':out=out[(out[k]==v)|out.method.str.endswith('_np')]
        else:out=out[out[k]==v]
    return out


def ordered(f,high=False):
    return [m for m in (HIGH if high else LOW) if m in set(f.method)]


def decorate(ax,index,title):
    ax.set_title(title,loc='left',pad=5)
    ax.text(-.16,1.03,string.ascii_lowercase[index],transform=ax.transAxes,fontweight='bold',fontsize=9)
    ax.tick_params(width=.7,length=3)


def legend(fig,methods,top=.94):
    handles=[Patch(facecolor=COLORS[m],edgecolor='#333333',hatch='//' if m.endswith('_np') else None,label=LABELS[m]) for m in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,top),ncol=min(4,len(handles)),columnspacing=1.1,handlelength=1.5)


def curve(ax,f,metric,x,methods):
    for method in methods:
        g=f[f.method==method].groupby(x)[metric].agg(['mean','std','count']).sort_index()
        if g.empty:continue
        ax.errorbar(g.index,g['mean'],yerr=g['std'].fillna(0)/np.sqrt(g['count']),
                    color=COLORS[method],marker='s' if method.endswith('_np') else 'o',markersize=3,
                    linestyle='--' if method.endswith('_np') else '-',linewidth=1,capsize=2)
    ax.grid(axis='y',color='#dddddd',linewidth=.45)
    if f.empty:ax.text(.5,.5,'Not available',ha='center',transform=ax.transAxes)


def boxes(ax,f,metric,group,groups,methods):
    width=.78/max(1,len(methods))
    for j,method in enumerate(methods):
        vals=[];positions=[]
        for i,g in enumerate(groups):
            values=f[(f[group]==g)&(f.method==method)][metric].dropna().to_numpy()
            if values.size:vals.append(values);positions.append(i-.39+(j+.5)*width)
        if not vals:continue
        artists=ax.boxplot(vals,positions=positions,widths=width*.85,patch_artist=True,
                           manage_ticks=False,showfliers=True,whis=1.5,
                           medianprops={'color':'black','linewidth':.7},
                           boxprops={'linewidth':.6},whiskerprops={'linewidth':.6},capprops={'linewidth':.6},
                           flierprops={'marker':'.','markersize':1.4,'alpha':.6,'markeredgecolor':COLORS[method]})
        for patch in artists['boxes']:
            patch.set_facecolor(COLORS[method]);patch.set_alpha(.85)
            if method.endswith('_np'):patch.set_hatch('//')
    ax.set_xticks(range(len(groups)),[str(g) for g in groups])
    ax.set_xlim(-.6,len(groups)-.4)
    for i in range(len(groups)-1):ax.axvline(i+.5,color='#b0b0b0',ls='--',lw=.6)
    if f.empty:ax.text(.5,.5,'Not available',ha='center',transform=ax.transAxes)


def save(fig,out,name,source,panels,caption,diagnostic):
    # Diagnostic previews contain actual smoke-fit values, never simulated plot placeholders.
    if diagnostic:fig.text(.5,.005,'DIAGNOSTIC / INCOMPLETE GRID — NOT FORMAL RESULTS',ha='center',fontsize=7,color='#9c2626')
    fig.savefig(out/f'{name}.pdf')
    fig.savefig(out/f'{name}.svg')
    fig.savefig(out/f'{name}.png',dpi=300)
    plt.close(fig)
    source.to_csv(out/f'{name}_source.csv',index=False)
    counts=source.groupby(['method','status'],dropna=False).size().reset_index(name='count').to_dict('records')
    atomic_json(out/f'{name}_panels.json',dict(panels=panels,source_counts=counts,
                diagnostic=diagnostic,excluded_observations='Only documented panel conditions; no outlier trimming'))
    (out/f'{name}_caption.txt').write_text(caption+'\n'+
        'Boxes: median, quartiles, 1.5-IQR whiskers, all outliers shown. Curves: mean +/- one MCSE. '+
        'nonDP fits are shared across privacy-budget columns, not independent repeated fits.\n'+
        ('Diagnostic/incomplete grid; not a completed result figure.\n' if diagnostic else ''))


def real_grid(f,out,ds,diagnostic):
    f=select(f,regime=ds,role='prediction');good=f[f.status=='ok']
    if f.empty:return
    methods=ordered(f,ds=='crime');fig,axs=plt.subplots(3,3,figsize=(7.2,6.9),sharey='row',squeeze=False)
    fig.subplots_adjust(left=.10,right=.985,bottom=.085,top=.84,wspace=.15,hspace=.39)
    title='Housing' if ds=='housing' else 'Communities and Crime'
    fig.suptitle(title+': quantile prediction',fontsize=11,y=.99)
    legend(fig,methods,top=.96);panels=[]
    groups=sorted(f.n_train.unique())
    for r,tau in enumerate([.25,.5,.75]):
        for c,eps in enumerate([.3,.5,.9]):
            ax=axs[r,c];data=select(good,tau=tau,epsilon=eps)
            active=[m for m in methods if not (tau!=.5 and (m.startswith('huber') or m in ['frappe','cai_iht']))]
            boxes(ax,data,'ql_tau','n_train',groups,active)
            decorate(ax,r*3+c,rf'$\tau={tau:g}$, $\epsilon={eps:g}$')
            if c==0:ax.set_ylabel(r'$\mathrm{QL}_{\tau}$')
            if r==2:ax.set_xlabel('Training sample size')
            ax.tick_params(axis='x',labelrotation=35)
            panels.append(dict(index=r*3+c,tau=tau,epsilon=eps,n_train=list(map(int,groups)),methods=active))
    save(fig,out,f'real_{ds}_quantile_grid',f,panels,
         'Test-set mean quantile loss across paired 80/20 splits and mechanism randomization. '+
         'Response units: log house value or original crime rate. Rows: tau; columns: epsilon. '+
         'Mean-target comparators appear only at tau=0.5.',diagnostic)


def low_overview(f,out,diagnostic):
    low=select(f,regime='low',p=10,tau=.5)
    if low.empty and diagnostic:
        low=select(f,regime='low',tau=.5)
    if low.empty:return
    methods=ordered(low);fig,axs=plt.subplots(2,3,figsize=(7.2,4.8))
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.80,wspace=.28,hspace=.48)
    fig.suptitle('Low-dimensional estimation',fontsize=11,y=.99);legend(fig,methods,top=.96)
    ns=sorted(low.n_train.unique());epsilons=sorted(low.epsilon.dropna().unique())
    target_eps=.5 if .5 in epsilons else epsilons[0]
    panels=[];used=[]
    for j,noise in enumerate(NOISES):
        for r in [0,1]:
            x='n_train' if r==0 else 'epsilon'
            data=select(low,noise=noise,**({'epsilon':target_eps} if r==0 else {'n_train':max(ns)}))
            # Expand the single nonDP result onto the public epsilon axis for display only.
            plotted=data
            if r==1:
                npdata=data[data.method.str.endswith('_np')]
                plotted=pd.concat([data[~data.method.str.endswith('_np')],*[npdata.assign(epsilon=e) for e in epsilons]],ignore_index=True)
            curve(axs[r,j],plotted[plotted.status=='ok'],'l2_error',x,methods)
            decorate(axs[r,j],r*3+j,NOISE_LABELS[noise]+(rf' | $\epsilon={target_eps:g}$' if r==0 else f' | n={max(ns)}'))
            axs[r,j].set_xlabel('Sample size n' if r==0 else r'Privacy budget $\epsilon$')
            if j==0:axs[r,j].set_ylabel(r'$\|\widehat\beta-\beta^*\|_2$')
            panels.append(dict(index=r*3+j,noise=noise,x=x));used.append(data)
    save(fig,out,'sim_low_overview',pd.concat(used).drop_duplicates('task_id'),panels,
         'Low-dimensional parameter error at tau=0.5. Huber is included in the Cauchy empirical stress test.',diagnostic)


def quantile_boxes(f,out,regime,diagnostic):
    low=regime=='low';part=select(f,regime=regime)
    if part.empty:return
    if low:
        target_p=10 if 10 in part.p.unique() else part.p.iloc[0]
        part=select(part,p=target_p,n_train=part.n_train.max())
    else:part=select(part,n_train=part.n_train.max(),true_sparsity=6 if 6 in part.true_sparsity.unique() else part.true_sparsity.iloc[0])
    noise_order=['gaussian','t3' if low else 't2.25','cauchy']
    epsilons=[e for e in ([.3,.9] if low else [.75,1.]) if e in part.epsilon.dropna().unique()]
    if not epsilons:epsilons=sorted(part.epsilon.dropna().unique())[-1:]
    methods=ordered(part,not low)
    for eps in epsilons:
        fig,axs=plt.subplots(1 if low else 2,3,figsize=(7.2,3.4 if low else 5.),squeeze=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.16 if low else .10,top=.74 if low else .80,wspace=.3,hspace=.45)
        fig.suptitle(('Low-dimensional quantiles' if low else 'High-dimensional robustness')+rf' | $\epsilon={eps:g}$',fontsize=11,y=.99)
        legend(fig,methods,top=.95);used=[];panels=[]
        for j,noise in enumerate(noise_order):
            data=select(part,noise=noise,epsilon=eps);used.append(data)
            for r,metric in enumerate(['l2_error'] if low else ['l2_error','recall']):
                ax=axs[r,j];boxes(ax,data[data.status=='ok'],metric,'tau',[.25,.5,.75],methods)
                decorate(ax,r*3+j,NOISE_LABELS[noise]);ax.set_xlabel(r'Quantile level $\tau$')
                if j==0:ax.set_ylabel('Parameter L2 error' if metric=='l2_error' else 'Recall')
                if metric=='recall':ax.set_ylim(-.03,1.03)
                panels.append(dict(index=r*3+j,noise=noise,metric=metric,epsilon=eps))
        save(fig,out,f'sim_{regime}_quantiles_eps{eps:g}',pd.concat(used),panels,
             'Parameter error'+(' and slope-support Recall' if not low else '')+' across quantiles; Huber at the Cauchy median is an empirical comparison.',diagnostic)


def high_core(f,out,diagnostic):
    part=select(f,regime='high',noise='gaussian',tau=.5)
    if part.empty:return
    if 'true_sparsity' in part and 6 in part.true_sparsity.unique():part=select(part,true_sparsity=6)
    epsilons=sorted(part.epsilon.dropna().unique());methods=ordered(part,True)
    fig,axs=plt.subplots(2,len(epsilons),figsize=(7.2,4.8),squeeze=False)
    fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.80,wspace=.28,hspace=.45)
    fig.suptitle('High-dimensional estimation and Recall',fontsize=11,y=.99);legend(fig,methods,top=.96)
    panels=[]
    for j,eps in enumerate(epsilons):
        data=select(part,epsilon=eps)
        for r,metric in enumerate(['l2_error','recall']):
            ax=axs[r,j];curve(ax,data[data.status=='ok'],metric,'n_train',methods)
            decorate(ax,r*len(epsilons)+j,rf'$\epsilon={eps:g}$');ax.set_xlabel('Sample size n')
            if j==0:ax.set_ylabel('Parameter L2 error' if r==0 else 'Recall')
            if r==1:ax.set_ylim(-.03,1.03)
            panels.append(dict(index=r*len(epsilons)+j,epsilon=eps,metric=metric))
    save(fig,out,'sim_high_core',part,panels,'Gaussian median core grid. Recall excludes the intercept.',diagnostic)


def supplements(f,out,diagnostic):
    low=select(f,regime='low',tau=.5,epsilon=.5)
    if not low.empty:
        ps=sorted(low.p.unique());fig,axs=plt.subplots(len(ps),3,figsize=(7.2,2.1*len(ps)+1.1),squeeze=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.1,top=.83,wspace=.28,hspace=.5)
        fig.suptitle('Low-dimensional dimension comparison',fontsize=11,y=.99);legend(fig,ordered(low),top=.95)
        for i,p in enumerate(ps):
            for j,noise in enumerate(NOISES):
                ax=axs[i,j];data=select(low,p=p,noise=noise)
                curve(ax,data[data.status=='ok'],'l2_error','n_train',ordered(low))
                decorate(ax,i*3+j,f'p={p} | '+NOISE_LABELS[noise]);ax.set_xlabel('Sample size n')
                if j==0:ax.set_ylabel('Parameter L2 error')
        save(fig,out,'sim_low_dimensions',low,[],r'Dimension supplement at tau=0.5 and epsilon=0.5.',diagnostic)
    high=select(f,regime='high',tau=.5,noise='gaussian')
    if not high.empty and high.true_sparsity.nunique()>1:
        high=select(high,n_train=high.n_train.max()).copy();high['active_slopes']=high.true_sparsity-1
        epsilons=[e for e in [.75,1.] if e in high.epsilon.dropna().unique()]
        fig,axs=plt.subplots(2,len(epsilons),figsize=(7.2,4.7),squeeze=False)
        fig.subplots_adjust(left=.09,right=.985,bottom=.10,top=.80,wspace=.28,hspace=.45)
        fig.suptitle('High-dimensional support-size comparison',fontsize=11,y=.99);legend(fig,ordered(high,True),top=.96)
        for j,e in enumerate(epsilons):
            data=select(high,epsilon=e)
            for i,metric in enumerate(['l2_error','recall']):
                curve(axs[i,j],data[data.status=='ok'],metric,'active_slopes',ordered(high,True))
                decorate(axs[i,j],i*len(epsilons)+j,rf'$\epsilon={e:g}$');axs[i,j].set_xlabel('Active slopes')
                if j==0:axs[i,j].set_ylabel('Parameter L2 error' if i==0 else 'Recall')
        save(fig,out,'sim_high_support_size',high,[],r'Support-size supplement at tau=0.5; working dimension remains 12.',diagnostic)


def build(inputs,output,allow_partial=False):
    setup();frames=[];signatures=set();diagnostic=False;provenance=[]
    for item in inputs:
        root=Path(item);manifest=json.loads((root/'manifest.json').read_text())
        spec=json.loads((root/'run_spec.json').read_text());signatures.add(spec['source_digest'])
        partial=not manifest['complete'] or not manifest['full_grid'] or manifest['n_failed']>0 or manifest['plan']['diagnostic_only']
        if partial and not allow_partial:raise ValueError(f'{root.name}: incomplete/diagnostic/failed fits; --allow-partial required')
        diagnostic|=partial;frame=pd.read_csv(root/'raw.csv');frames.append(frame)
        provenance.append(dict(input=str(root),fingerprint=spec['fingerprint'],n_failed=manifest['n_failed']))
    if len(signatures)!=1:raise ValueError('Figure inputs come from different source versions')
    f=pd.concat(frames,ignore_index=True)
    identity=[x for x in ['study','regime','n_train','p','true_sparsity','noise','tau','epsilon','method','rep','role'] if x in f]
    if f.duplicated(identity).any():raise ValueError('Overlapping fits in figure inputs; remove duplicate run directories')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Never retain old figures from another call in the same destination.
    if (out/'figure_manifest.json').exists():
        old=json.loads((out/'figure_manifest.json').read_text())
        if old['inputs']!=provenance:raise ValueError('Figure destination belongs to different inputs; choose a new directory')
    real_grid(f,out,'housing',diagnostic);real_grid(f,out,'crime',diagnostic)
    low_overview(f,out,diagnostic);high_core(f,out,diagnostic)
    quantile_boxes(f,out,'low',diagnostic);quantile_boxes(f,out,'high',diagnostic);supplements(f,out,diagnostic)
    # Only the retained supplemental metrics, with success/failure counts.
    metrics=[m for m in ['excess_ql','support_coverage'] if m in f]
    for metric in metrics:
        good=f[(f.status=='ok')&f[metric].notna()]
        keys=[k for k in ['regime','n_train','p','true_sparsity','noise','tau','epsilon','method'] if k in good]
        tab=good.groupby(keys,dropna=False)[metric].agg(['mean','std','count']).reset_index()
        tab['mcse']=tab['std']/np.sqrt(tab['count']);tab.to_csv(out/f'supp_{metric}.csv',index=False)
        # Compact representative manuscript table, complete CSV retained above.
        mask=(tab.regime=='low')&(tab.n_train==1000)&(tab.p==10) if metric=='excess_ql' else tab.regime=='high'
        if metric=='excess_ql':mask=mask|((tab.regime=='high')&(tab.n_train==10000)&(tab.true_sparsity==6))
        compact=tab[mask]
        if not compact.empty:compact.to_latex(out/f'supp_{metric}.tex',index=False,float_format='%.4f')
    f[f.status!='ok'].to_csv(out/'failed_fits.csv',index=False)
    for item in inputs:
        root=Path(item)
        for name in ['coefficients_raw.csv','coefficients_summary.csv','coefficients_table.tex']:
            if (root/name).exists():shutil.copy2(root/name,out/('real_housing_'+name))
    atomic_json(out/'figure_manifest.json',dict(inputs=provenance,diagnostic=diagnostic,
                source_digest=next(iter(signatures)),n_source_rows=len(f),
                outputs=sorted(p.name for p in out.iterdir() if p.is_file())))
    print(f'Figures and source-data tables saved in {out}')
