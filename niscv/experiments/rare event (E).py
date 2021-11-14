import numpy as np
import scipy.stats as st
from niscv.clustering.probability import Probability
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(dim, b, size_est, show, size_kn, ratio, resample=True, auto=False, k=None):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    indicator = lambda x: 1 * (x[:, 0] > b)
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    pro = Probability(dim, target, indicator, init_proposal, size_est, show=show)
    pro.initial_estimation(size_kn, ratio, resample=resample)
    if resample:
        results.extend([pro.result[-5], pro.result[-4]])
    else:
        results.extend([pro.result[-2], pro.result[-1]])

    if pro.show:
        pro.draw(grid_x, name='initial')

    pro.density_estimation(bw=1.0, auto=auto, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1, k=k)
    pro.nonparametric_estimation()
    results.extend([pro.result[-4], pro.result[-3], pro.result[-2], pro.result[-1]])
    if pro.show:
        pro.draw(grid_x, name='nonparametric')

    pro.regression_estimation()
    results.extend([pro.result[-2], pro.result[-1]])
    if pro.show:
        pro.draw(grid_x, name='regression')

    if resample:
        pro.likelihood_estimation(optimize=True, NR=True)
        results.append(pro.result[-1])

    return results, pro.result


def run(it, b):
    np.random.seed(1997 * it + 1107)
    print(it, end=' ')
    settings = [[True, 1], [False, None], [True, 2], [True, 3], [True, 4]]
    ratios = [5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 500, 1000]

    result, result_all = experiment(dim=5, b=b, size_est=10000, show=False, size_kn=500, ratio=1000, resample=False,
                                    auto=True, k=1)
    Results = [result]
    Results_all = [result_all]
    for setting in settings:
        results = []
        results_all = []
        for ratio in ratios:
            result, result_all = experiment(dim=5, b=b, size_est=10000, show=False, size_kn=500, ratio=ratio,
                                            resample=True, auto=setting[0], k=setting[1])
            results.append(result)
            results_all.append(result_all)

        Results.append(results)
        Results_all.append(results_all)

    return [Results, Results_all]


def main(b):
    os.environ['OMP_NUM_THREADS'] = '2'
    with multiprocessing.Pool(processes=16) as pool:
        begin = dt.now()
        its = np.arange(100)
        R = pool.map(partial(run, b=b), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('rare_(' + str(b) + ')', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    # main(2.0)
    R1 = experiment(dim=5, b=2.0, size_est=10000, show=False, size_kn=500, ratio=1000,
                    resample=False, auto=True, k=1)
    R2 = experiment(dim=5, b=2.0, size_est=10000, show=False, size_kn=500, ratio=1000,
                    resample=True, auto=True, k=1)
    R3 = experiment(dim=5, b=2.0, size_est=10000, show=False, size_kn=500, ratio=1000,
                     resample=True, auto=True, k=2)
    print(R1)
    print(R2)
    print(R3)
