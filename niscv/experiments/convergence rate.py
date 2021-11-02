import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from niscv.basic.expectation import Expectation
import sklearn.linear_model as lm
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
    results.extend([exp.result[-4]])
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


def main(order, sn, size_kns):
    results = []
    for size_kn in size_kns:
        print(size_kn)
        begin = dt.now()
        result = experiment(dim=5, order=order, size_est=100000, sn=sn, show=False, size_kn=size_kn, ratio=100)
        end = dt.now()
        result = np.append(result, (end - begin).seconds)
        results.append(result)

    return np.array(results)


if __name__ == '__main__':
    settings = [[1, False], [1, True], [2, False], [2, True]]
    X = np.array([50, 100, 150, 200, 300, 400, 500, 700, 900, 1100])
    for setting in settings:
        R = main(order=setting[0], sn=setting[1], size_kns=X)

        plt.loglog(X, R[:, 0], label='IS')
        plt.loglog(X, R[:, 1], label='NIS')
        plt.loglog(X, R[:, 2], label='MIS')
        plt.loglog(X, R[:, 3], label='RIS')
        plt.legend()
        plt.show()

        plt.loglog(X, R[:, 4], label='time')
        plt.show()

        reg1 = lm.LinearRegression().fit(np.log(X).reshape([-1, 1]), np.log(R[:, 1]))
        reg2 = lm.LinearRegression().fit(np.log(X).reshape([-1, 1]), np.log(R[:, 2]))
        reg3 = lm.LinearRegression().fit(np.log(X).reshape([-1, 1]), np.log(R[:, 3]))
        reg4 = lm.LinearRegression().fit(np.log(X).reshape([-1, 1]), np.log(R[:, 4] + 1e-10))
        print(setting, reg1.coef_, reg2.coef_, reg3.coef_)
        print(reg4.coef_)
