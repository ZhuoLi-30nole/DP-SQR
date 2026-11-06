import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from dpsqr.policy import total_delta, allocation, parameters, methods_for, FRACTIONS
from dpsqr.privacy import rho_from_eps_delta,eps_from_rho_delta
from dpsqr.baselines.huber import huber_iht,highdim_step_budget
from dpsqr.initialization import initialize_huber,_smooth_sqr_objective
from dpsqr.pipeline import fit_pipeline
from dpsqr.study import prepare_real,original_coefficients,lad_fit,plan,validate_config

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_all_configs_and_counts(self):
        for path in (ROOT/'configs').glob('*.json'):
            cfg=json.loads(path.read_text());self.assertGreater(plan(cfg)['complete_fits'],0)
        self.assertEqual(plan(json.loads((ROOT/'configs/sim_high_core.json').read_text()))['scenario_jobs'],900)

    def test_stage_composition_and_tran_cap(self):
        for n in [48,250,400,1000,10000,16000]:
            for e in [.1,.3,.5,.9,1.]:
                for name in FRACTIONS:
                    a=allocation(name,e,total_delta(n),n)
                    self.assertAlmostEqual(a['eps_s']+a['eps_i']+a['eps_m'],e)
                    self.assertLessEqual(a['del_s']+a['del_i']+a['del_m'],total_delta(n)+1e-12)
                    if name=='tran_dpsgd':self.assertLessEqual(a['del_m'],1/n)

    def test_cauchy_huber_included_but_no_nonmedian_huber(self):
        for regime in ['low','high']:
            names=methods_for(regime,.5,'cauchy')
            self.assertTrue(any(x.startswith('huber') for x in names))
            self.assertFalse(any(x.startswith('huber') for x in methods_for(regime,.25,'cauchy')))
        self.assertNotIn('cai_iht',methods_for('high',.5,'cauchy'))

    def test_small_or_filtered_runs_cannot_masquerade_as_formal(self):
        cfg=json.loads((ROOT/'configs/real_housing.json').read_text())
        self.assertFalse(plan(cfg)['diagnostic_only'])
        cfg['repeats']=2
        self.assertTrue(plan(cfg)['diagnostic_only'])
        cfg['repeats']=300;cfg['methods']=['sqr_gd']
        self.assertTrue(plan(cfg)['diagnostic_only'])

    def test_sqr_objective_gradient_uses_actual_quantile(self):
        rng=np.random.default_rng(4);Z=rng.normal(size=(25,4));y=rng.normal(size=25);b=rng.normal(size=4)
        for tau in [.25,.5,.75]:
            value,g=_smooth_sqr_objective(b,Z,y,tau,.6,.2,'gaussian')
            finite=np.array([(_smooth_sqr_objective(b+np.eye(4)[j]*1e-6,Z,y,tau,.6,.2,'gaussian')[0]-
                              _smooth_sqr_objective(b-np.eye(4)[j]*1e-6,Z,y,tau,.6,.2,'gaussian')[0])/2e-6 for j in range(4)])
            np.testing.assert_allclose(g,finite,atol=1e-8)

    def test_no_artificial_ten_ball_for_huber_initialization(self):
        result=initialize_huber(np.zeros((60,1)),np.full(60,50.),epsilon_initialization=.01,
                                delta_initialization=.001,output_radius=None,private=True,
                                rng=np.random.default_rng(2))
        self.assertIsNone(result.output_radius)
        self.assertGreater(np.linalg.norm(result.beta),10.)
        X=np.zeros((100,1));y=np.r_[np.full(50,-30.),np.full(50,60.)]
        result=initialize_huber(X,y,epsilon_initialization=.2,delta_initialization=.001,
                                output_radius=None,private=False)
        self.assertIsNone(result.output_radius)
        self.assertLessEqual(result.metadata['uniform_gap_bound'],1e-8)

    def test_s5_huber_accountant_and_execution(self):
        for n in [48,250,400,10000]:
            for e in [.3,.5,.9]:
                a=allocation('huber_iht',e,total_delta(n),n);T=int(np.ceil(3*np.log(n)))
                et,dt=highdim_step_budget(T,a['eps_m'],a['del_m'])
                rho=T*et**2/(8*np.log(1/dt))
                self.assertLessEqual(eps_from_rho_delta(rho,a['del_m']),a['eps_m'])
        rng=np.random.default_rng(1);X=np.column_stack([np.ones(60),rng.normal(size=(60,8))]);y=rng.normal(size=60)
        p=parameters('huber_iht',n=60,p=9,regime='crime',s=5,intercept=True)
        fit=fit_pipeline('huber_iht',X,y,p,epsilon=.9,delta=total_delta(60),tau=.5,seed=3)
        self.assertLessEqual(np.count_nonzero(fit.coef),5)
        self.assertIn('rho_main_bound',fit.privacy['main'])

    def test_quantile_noise_is_scaled_for_tail_quantiles(self):
        X=np.linspace(-1,1,60)[:,None];y=2*X[:,0]
        values=[]
        for tau in [.25,.5,.75]:
            p=parameters('sqr_gd',n=60,p=1,regime='low')
            fit=fit_pipeline('sqr_gd',X,y,p,epsilon=.9,delta=total_delta(60),tau=tau,seed=3)
            values.append(fit.privacy['main']['gradient_sd'])
        self.assertAlmostEqual(values[0]/values[1],1.5)
        self.assertAlmostEqual(values[2]/values[1],1.5)

    def test_test_rows_do_not_change_training_scaler(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(100,3));y=rng.normal(size=100)
        c=dict(seed=9,regime='housing');t=dict(sample_m=80,n=64,rep=2)
        with patch('dpsqr.study.benchmark',return_value=(X,y,{'predictors':['a','b','c']})):
            first=prepare_real(c,t)
        from dpsqr.study import derived_seed
        ids=np.random.default_rng(derived_seed(9,'records-and-split','housing',80,2)).permutation(100)[:80]
        changed=X.copy();changed[ids[64:]]+=100000
        changed_y=y.copy();changed_y[ids[64:]]-=100000
        with patch('dpsqr.study.benchmark',return_value=(changed,changed_y,{'predictors':['a','b','c']})):
            second=prepare_real(c,t)
        np.testing.assert_array_equal(first[0],second[0]);np.testing.assert_array_equal(first[1],second[1])
        self.assertTrue(first[-1]['split_disjoint'])

    def test_inverse_transform_predictions(self):
        X=np.array([[2.,4.],[5.,7.]])
        transform=dict(x_mean=np.array([1.,3.]),x_sd=np.array([2.,4.]),y_mean=10.,y_sd=3.)
        b=np.array([.4,1.2,-.6]);orig=original_coefficients(b,transform)
        expected=10+3*np.column_stack([np.ones(2),(X-transform['x_mean'])/transform['x_sd']])@b
        np.testing.assert_allclose(np.column_stack([np.ones(2),X])@orig,expected)

    def test_lad_resists_response_outlier(self):
        X=np.column_stack([np.ones(20),np.arange(20.)]);y=1+2*X[:,1];y[-1]+=1000
        b=lad_fit(X,y);np.testing.assert_allclose(b,[1.,2.],atol=1e-6)

    def test_simulation_runner_end_to_end(self):
        from dpsqr.study import run_task
        cfg=json.loads((ROOT/'configs/smoke_sim_low.json').read_text())
        cfg['methods']=['sqr_gd','sqr_gd_np'];cfg['epsilon']=[.9]
        task=dict(n=60,p=5,true_sparsity=5,tau=.25,noise='cauchy',rep=0)
        with tempfile.TemporaryDirectory() as out:
            rows=run_task((cfg,task,out))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['status']=='ok' for r in rows),rows)
            # A resumed task must reuse exactly the saved fit, including timing.
            self.assertEqual(rows,run_task((cfg,task,out)))


if __name__=='__main__':unittest.main()
