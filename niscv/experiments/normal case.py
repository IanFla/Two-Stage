import numpy as np
# from matplotlib import pyplot as plt
import scipy.stats as st
from niscv.basic.expectation import Expectation
# import sklearn.linear_model as lm
from datetime import datetime as dt
import pickle


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

    exp.likelihood_estimation(optimize=True, NR=True)
    results.extend([exp.result[-1], results[-1]])
    return results,  exp.result


def main(rep, order, sn, size_kns):
    results = []
    results_all = []
    for size_kn in size_kns:
        begin = dt.now()
        result = []
        result_all = []
        for i in range(rep):
            res, res_all = experiment(dim=5, order=order, size_est=100000, sn=sn,
                                      show=False, size_kn=size_kn, ratio=100)
            result.append(res)
            result_all.append(res_all)

        end = dt.now()
        print((end - begin).seconds)
        results.append(result)
        results_all.append(result_all)

    return np.array(results), results_all


if __name__ == '__main__':
    np.random.seed(19971107)
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    X = np.array([100, 150, 200, 300, 400, 600, 800])
    R = []
    R_all = []
    for setting in settings:
        r, r_all = main(rep=100, order=setting[0], sn=setting[1], size_kns=X)
        R.append(R)
        R_all.append(r_all)

    with open('DimSize', 'wb') as file:
        pickle.dump([R, R_all], file)
        file.close()
