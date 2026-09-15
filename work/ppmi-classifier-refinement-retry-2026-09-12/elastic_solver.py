"""Convex elastic-net logistic regression using nonnegative coefficient splitting."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


class ElasticLogistic:
    """Same unweighted, unpenalized-intercept objective as sklearn SAGA.

    mean(log(1+exp(Xw+b)) - y*(Xw+b))
      + (1-ratio)/(2*n*C)*||w||^2 + ratio/(n*C)*||w||_1.
    Write w = positive - negative, with both components constrained >= 0.
    This gives a smooth convex bound-constrained objective for L-BFGS-B.
    """

    def __init__(self, c, ratio):
        self.c, self.ratio = c, ratio

    def fit(self, x, y):
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = x.shape
        assert set(np.unique(y)) == {0., 1.} and 0 < self.ratio < 1 and self.c > 0
        l1, l2 = self.ratio / (n * self.c), (1 - self.ratio) / (n * self.c)

        def objective(v):
            w, b = v[:p] - v[p:2*p], v[-1]
            z = x @ w + b
            residual = (expit(z) - y) / n
            value = np.mean(np.logaddexp(0, z) - y * z) + .5 * l2 * (w @ w) + l1 * v[:2*p].sum()
            gradient = x.T @ residual + l2 * w
            return value, np.r_[gradient + l1, -gradient + l1, residual.sum()]

        v = np.zeros(2*p + 1)
        v[-1] = logit(y.mean())
        bounds = [(0., None)] * (2*p) + [(None, None)]
        total = 0
        for tolerance in [1e-14, 1e-16]:
            result = minimize(objective, v, jac=True, bounds=bounds, method='L-BFGS-B',
                              options=dict(maxiter=10000, maxfun=30000, maxls=50, maxcor=20,
                                           ftol=tolerance, gtol=1e-8))
            v = result.x
            total += result.nit
            w, b = v[:p] - v[p:2*p], v[-1]
            # Independently check the original nonsmooth problem's stationarity.
            residual = (expit(x @ w + b) - y) / n
            gradient = x.T @ residual + l2 * w
            kkt = np.where(w != 0, np.abs(gradient + l1 * np.sign(w)), np.maximum(np.abs(gradient) - l1, 0))
            error = max(float(kkt.max()), abs(float(residual.sum())))
            if error <= 5e-7:
                self.coef_, self.intercept_ = w[None, :], np.array([b])
                self.classes_, self.n_iter_ = np.array([0, 1]), np.array([total])
                self.total_optimizer_iterations_ = total
                self.kkt_residual_ = error
                self.objective_ = float(objective(v)[0])
                self.optimizer_message_ = str(result.message)
                assert np.isfinite(w).all() and np.isfinite(b)
                return self
        raise RuntimeError(f'Elastic-net KKT check failed: residual={error}, C={self.c}, ratio={self.ratio}, p={p}')

    def predict_proba(self, x):
        p = expit(np.asarray(x) @ self.coef_[0] + self.intercept_[0])
        return np.column_stack([1 - p, p])


def validate_solver():
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    rng = np.random.default_rng(20260912)
    x = rng.normal(size=(120, 25))
    x[:, 3] = x[:, 2] + rng.normal(0, .02, 120)
    y = rng.binomial(1, expit(x[:, 0] - .7*x[:, 1] + .4))
    comparisons = []
    for c in [.01, .1, 1.]:
        for ratio in [.1, .5, .9]:
            fitted = ElasticLogistic(c, ratio).fit(x, y)
            reference = LogisticRegression(C=c, l1_ratio=ratio, penalty='elasticnet', solver='saga',
                                           tol=1e-10, max_iter=100000, random_state=20260912)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                reference.fit(x, y)
            np.testing.assert_allclose(fitted.coef_, reference.coef_, atol=1e-4, rtol=1e-4)
            if np.count_nonzero(reference.coef_) == 0:
                # SAGA can stop when all coefficients stay zero before its intercept
                # reaches the optimum. The exact constant-model solution is prevalence.
                assert np.count_nonzero(fitted.coef_) == 0
                np.testing.assert_allclose(fitted.predict_proba(x)[:, 1], y.mean(), atol=1e-10)
                comparison = 'Exact intercept-only optimum; SAGA zero-coefficient stopping differs'
            else:
                np.testing.assert_allclose(fitted.predict_proba(x), reference.predict_proba(x), atol=1e-5, rtol=1e-5)
                comparison = 'Converged SAGA coefficient/probability agreement'
            comparisons.append(dict(C=c, l1_ratio=ratio, kkt_residual=fitted.kkt_residual_,
                                    comparison=comparison,
                                    maximum_probability_difference=float(np.max(np.abs(fitted.predict_proba(x)-reference.predict_proba(x))))))
    return comparisons
