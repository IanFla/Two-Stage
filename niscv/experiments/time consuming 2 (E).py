import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation
from datetime import datetime as dt
import pickle


def experiment(dim, order, size_est, sn, show, size_kn, ratio):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** order + 1
    init_proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)

    start = dt.now()
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=show)
    exp.initial_estimation(size_kn, ratio, resample=True)
    end1 = dt.now()
    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    end2 = dt.now()
    exp.regression_estimation()
    end3 = dt.now()
    exp.likelihood_estimation()
    end4 = dt.now()
    return [end1 - start, end2 - end1, end3 - end2, end4 - end3]


def run(it, dim):
    print(dim, it)
    np.random.seed(1997 * it + 1107)
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    size_ests = [1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 70000, 100000]

    Results = []
    for setting in settings:
        print(setting)
        results = []
        for size_est in size_ests:
            result = experiment(dim=dim, order=setting[0], size_est=size_est, sn=setting[1],
                                            show=False, size_kn=300, ratio=1000)
            results.append(result)

        Results.append(results)

    return Results


def main(dim):
    R = []
    for i in range(10):
        R.append(run(i, dim))

    with open('time2_' + str(dim) + 'D', 'wb') as file:
        pickle.dump(R, file)

    return R


if __name__ == '__main__':
    R1 = main(4)
    R2 = main(6)
    R3 = main(8)
