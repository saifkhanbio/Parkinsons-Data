"""Positive-slope logistic calibration; no raw RNA or held-out labels accepted."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
MIN_SLOPE=1e-6

def apply(state,p):
    p=np.asarray(p,float)
    assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
    return expit(state['intercept']+state['slope']*logit(p))

def fit(p,y):
    p=np.asarray(p,float);y=np.asarray(y,float)
    assert p.shape==y.shape and p.ndim==1 and set(y)=={0.,1.}
    assert np.isfinite(p).all() and ((p>0)&(p<1)).all()
    z=logit(p);x=np.c_[np.ones(len(y)),z]
    def objective(theta):
        eta=x@theta
        return np.mean(np.logaddexp(0,eta)-y*eta),x.T@(expit(eta)-y)/len(y)
    r=minimize(objective,[logit(y.mean()),.25],jac=True,method='L-BFGS-B',bounds=[(None,None),(MIN_SLOPE,None)],options={'ftol':1e-14,'gtol':1e-10,'maxiter':2000,'maxls':50})
    loss,g=objective(r.x);projected=g.copy()
    if r.x[1]<=MIN_SLOPE+1e-10:projected[1]=min(0,g[1])
    assert r.success and np.max(abs(projected))<1e-7,(r.message,g)
    return dict(intercept=float(r.x[0]),slope=float(r.x[1]),slope_at_bound=bool(r.x[1]<=MIN_SLOPE+1e-10),training_n=len(y),training_log_loss=float(loss),projected_gradient_max=float(np.max(abs(projected))),optimizer_iterations=int(r.nit))
