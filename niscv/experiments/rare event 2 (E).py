import numpy as np
import scipy.stats as st
from niscv.clustering.probability import Probability
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(dim, b, size_est, show, size_kn, ratio, resample=True, mode=0):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    indicator = lambda x: 1 * (x[:, 0] >= b)
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

    pro.density_estimation(bw=1.0, mode=mode, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
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
    modes = [1, 0, 2, 3, 4, 5]

    results = []
    results_all = []
    for mode in modes:
        result, result_all = experiment(dim=5, b=b, size_est=10000, show=False, size_kn=500,
                                        ratio=1000, resample=True, mode=mode)
        results.append(result)
        results_all.append(result_all)

    return [results, results_all]


def main(b):
    os.environ['OMP_NUM_THREADS'] = '2'
    with multiprocessing.Pool(processes=16) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(partial(run, b=b), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('rare4_(' + str(b) + ')', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(0)
    main(1)
    main(3)
