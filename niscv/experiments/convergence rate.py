import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


def experiment(dim, order, size_est, sn, show, size_kn, ratio):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** order + 1
    init_proposal = st.multivariate_normal(mean=mean, cov=np.append(4, np.ones(dim - 1)))
    grid_x = np.linspace(-5, 5, 200)
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=show)
    exp.initial_estimation(size_kn, ratio, resample=True)
    results.append(exp.result[-5])
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.05)
    exp.nonparametric_estimation()
    results.extend([exp.result[-4], exp.result[-2]])
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    # exp.regression_estimation()
    # results.append(exp.result[-2])
    # if exp.show:
    #     exp.draw(grid_x, name='regression')

    return results,  exp.result


def run(it, dim):
    np.random.seed(1997 * it + 1107)
    print(it, end=' ')
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    # size_kns = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    size_kns = [100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000, 5000]

    Results = []
    Results_all = []
    for setting in settings:
        results = []
        results_all = []
        for size_kn in size_kns:
            result, result_all = experiment(dim=dim, order=setting[0], size_est=10 * size_kn, sn=setting[1],
                                            show=False, size_kn=size_kn, ratio=1000)
            results.append(result)
            results_all.append(result_all)

        Results.append(results)
        Results_all.append(results_all)

    return [Results, Results_all]


def main(dim):
    os.environ['OMP_NUM_THREADS'] = '2'
    with multiprocessing.Pool(processes=16) as pool:
        begin = dt.now()
        its = np.arange(100)
        R = pool.map(partial(run, dim=dim), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('convergence5_' + str(dim) + 'D', 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    main(3)
    main(6)
    main(9)
