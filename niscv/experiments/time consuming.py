import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation
from datetime import datetime as dt
import pickle


def experiment(dim, order, size_est, sn, show, size_kn, ratio):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** order + 1
    init_proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)

    print(0)
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=show)
    exp.initial_estimation(size_kn, ratio, resample=True)
    print(1)
    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    print(2)
    exp.regression_estimation()
    print(3)
    exp.likelihood_estimation()
    print(4)
    return results


def run(dim):
    np.random.seed(19971107)
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    size_kns = [50, 100, 150, 200, 300, 400, 500, 600]

    Results = []
    for setting in settings:
        results = []
        for size_kn in size_kns:
            result = experiment(dim=dim, order=setting[0], size_est=10000, sn=setting[1],
                                            show=False, size_kn=size_kn, ratio=1000)
            results.append(result)

        Results.append(results)

    return Results


def main(dim):
    experiment(dim, order=2, size_est=100000, sn=True, show=True, size_kn=500, ratio=1000)


if __name__ == '__main__':
    main(5)
