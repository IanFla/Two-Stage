import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation


def experiment(dim, size_est, sn, show, size_kn, ratio):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** 2
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=show)
    exp.initial_estimation(size_kn, ratio, resample=True)
    results.extend([exp.result[-3], exp.result[-2], exp.result[-4]])
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    results.extend([exp.result[-3], exp.result[-1]])
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.regression_estimation()
    results.extend([exp.result[-1]])
    if exp.show:
        exp.draw(grid_x, name='regression')

    exp.likelihood_estimation(optimize=True, NR=True)
    return results


def main(sn, ratios):
    # np.random.seed(3033079628)
    results = []
    for ratio in ratios:
        print(ratio)
        result = experiment(dim=4, size_est=25000, sn=sn, show=False, size_kn=500, ratio=ratio)
        results.append(result)

    return np.array(results)


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5, 7, 10, 15])
    R = main(sn=True, ratios=X)
    n0 = 500 * X
    ESS = R[:, 0]
    RSS = R[:, 1]
    print(np.round(n0 / ESS, 0))
    print(np.round(n0 / RSS, 0))
    print(np.round(R[:, 2:].T, 6))
