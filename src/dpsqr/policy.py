"""Shared, public experiment rules. No observed losses select parameters."""
import math

VERSION = "20261101-fixed-rules-v1"
FRACTIONS = {
    "sqr_gd": (0, 0, 1/12, 1/6, 11/12, 5/6),
    "huber_gd": (0, 0, 1/3, 1/6, 2/3, 5/6),
    "sqr_sr": (3/4, 1/3, 1/12, 1/3, 1/6, 1/3),
    "sqr_iht": (1/3, 0, 1/12, 1/2, 7/12, 1/2),
    "huber_iht": (1/3, 0, 1/3, 1/2, 1/3, 1/2),
    "tran_dpsgd": (0, 0, 0, 0, 1, 1),
    "tran_opm": (0, 0, 0, 0, 1, 0),
    "cai_iht": (0, 0, 0, 0, 1, 1),
    "frappe": (0, 0, 1/3, 1/3, 2/3, 2/3),
}
LOW = ["sqr_gd", "sqr_gd_np", "tran_dpsgd", "tran_opm", "huber_gd", "frappe"]
HIGH = ["sqr_sr", "sqr_sr_np", "sqr_iht", "sqr_iht_np", "huber_iht", "cai_iht"]


def total_delta(n):
    if not isinstance(n, int) or n < 10:
        raise ValueError("Sample size must be an integer >=10")
    return 10 * n ** -1.1


def allocation(method, epsilon, delta, n):
    """Stage budgets are in (epsilon,delta), not fractions of zCDP rho."""
    if not 0 < epsilon <= 1 or not 0 < delta < 1:
        raise ValueError("Expected 0<epsilon<=1 and 0<delta<1")
    f = FRACTIONS[method.removesuffix("_np")]
    e_s,d_s,e_i,d_i,e_m,d_m = f
    d_main = d_m * delta
    if method.removesuffix("_np") == "tran_dpsgd":
        d_main = min(d_main, 1/n)
    return dict(eps_s=e_s*epsilon, del_s=d_s*delta,
                eps_i=e_i*epsilon, del_i=d_i*delta,
                eps_m=e_m*epsilon, del_m=d_main)


def methods_for(regime, tau, noise=None):
    names = LOW if regime in ("low", "housing") else HIGH
    out=[]
    for name in names:
        if tau != .5 and (name.startswith("huber") or name in ("cai_iht", "frappe")):
            continue
        # Huber is deliberately retained for the median Cauchy stress test.
        if name == "cai_iht" and noise == "cauchy":
            continue
        out.append(name)
    return out


def parameters(method, *, n, p, regime, s=None, intercept=False):
    """Same rules for public simulations and public-data benchmarks.

    Steps in real-data fits are fixed empirical constants; no unknown
    population covariance bound or local-theorem condition is asserted.
    """
    base=method.removesuffix("_np")
    low=regime in ("low", "housing")
    T=math.ceil(3*math.log(n))
    if not low and (s is None or not 1 < s <= p):
        raise ValueError("Sparse experiments require public 1<s<=p")
    common={"initializer_mode":"uniform_gap", "initializer_radius":None}
    if intercept: common["intercept_index"]=0
    h=((p+math.log(n))/n)**.25 if low else (math.log(p*n)/n)**.25
    if base.startswith("sqr_"):
        opts=dict(h=h,kernel="gaussian",steps=T,
                  eta=.75 if low else .4,
                  gamma=math.sqrt(p) if low else math.sqrt(s) if base=="sqr_sr" else 1.)
        if not low: opts["size"]=s
    elif base.startswith("huber_"):
        opts=dict(T=T,lr=.2 if low else .01,
                  B=.5*math.sqrt(p+math.log(n)) if low else max(1.,.5*math.sqrt(math.log(p*n))))
        if not low: opts["s"]=s
    elif base=="cai_iht":
        opts=dict(sparsity=s,steps=T,step_size=2/15,x_bound=5.,coordinate_clip=5.,
                  response_clip=4*math.sqrt(math.log(n)),beta_radius=4.)
    elif base=="tran_dpsgd":
        opts=dict(h=.5,Cx=math.sqrt(p),kernel="logistic",preprocessing="l2",
                  step_schedule="paper_constant",parameter_radius=10.)
    elif base=="tran_opm":
        opts=dict(h=h,Cx=math.sqrt(p),kernel="logistic",preprocessing="l2",
                  mode="fixed",calibration="replace_one")
    elif base=="frappe":
        opts=dict(protocol="liu200",x_bound=math.sqrt(p),beta_radius=3.,
                  outer_steps=10,inner_steps=T,step_size=.9/p)
    else: raise ValueError(method)
    return {**common, **opts}
