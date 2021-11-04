import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation
import multiprocessing
import pickle
from datetime import datetime as dt


def experiment(dim, order, size_est, sn, show, size_kn, ratio):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** order
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=show)
    exp.initial_estimation(size_kn, ratio, resample=True)
    results.extend([exp.result[-5], exp.result[-4]])
    if exp.show:
        exp.draw(grid_x, name='initial')

    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    results.extend([exp.result[-4], exp.result[-3], exp.result[-2], exp.result[-1]])
    if exp.show:
        exp.draw(grid_x, name='nonparametric')

    exp.regression_estimation()
    results.extend([exp.result[-2], exp.result[-1]])
    if exp.show:
        exp.draw(grid_x, name='regression')

    return results,  exp.result


def main(it):
    np.random.seed(1997 * it + 1107)
    print(it)
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    Results = []
    Results_all = []
    for setting in settings:
        size_kns = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500]
        results = []
        results_all = []
        print(setting)
        for size_kn in size_kns:
            print(size_kn)
            result, result_all = experiment(dim=3, order=setting[0], size_est=20000, sn=setting[1],
                                            show=False, size_kn=size_kn, ratio=100)
            results.append(result)
            results_all.append(result_all)

        Results.append(results)
        Results_all.append(results_all)

    return [Results, Results_all]


if __name__ == '__main__':
    pool = multiprocessing.Pool(10)
    begin = dt.now()
    its = np.arange(200)
    R = pool.map(main, its)
    end = dt.now()
    print((end - begin).seconds)

    with open('normal_3D', 'wb') as file:
        pickle.dump(R, file)
        file.close()
