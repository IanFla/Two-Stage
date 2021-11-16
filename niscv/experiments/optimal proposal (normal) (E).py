import numpy as np
import scipy.stats as st
from niscv.basic.expectation import Expectation
from matplotlib import pyplot as plt


def experiment(dim, order, size_est, sn, size_kn, ratio, ax):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    fun = lambda x: x[:, 0] ** order + 1
    init_proposal = st.multivariate_normal(mean=mean + 0.5, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    grid_X = np.zeros([grid_x.size, dim])
    grid_X[:, 0] = grid_x

    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=True)
    exp.initial_estimation(size_kn, ratio, resample=True)
    opt = exp.opt_proposal(grid_X)
    init = exp.init_proposal(grid_X)
    ax.plot(grid_x, opt, c='k', label='optimal proposal')
    ax.plot(grid_x, opt.max() * init / init.max() / 10, c='b', label='initial proposal')
    results.extend([exp.result[-5], exp.result[-4]])
    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    nonpar = exp.nonpar_proposal(grid_X)
    mix = exp.mix_proposal(grid_X)
    ax.plot(grid_x, opt.max() * nonpar / nonpar.max(), c='y', label='nonparametric proposal')
    ax.plot(grid_x, opt.max() * mix / mix.max(), c='c', label='mixture proposal')
    results.extend([exp.result[-4], exp.result[-3], exp.result[-2], exp.result[-1]])
    exp.regression_estimation()
    reg = (exp.reg1.coef_ - exp.mu * exp.reg2.coef_).dot(exp.controls(grid_X)) \
        if exp.sn else exp.reg.coef_.dot(exp.controls(grid_X)) + exp.mu * exp.mix_proposal(grid_X)
    ax.plot(grid_x, np.abs(reg), c='r', label='regression proposal')
    results.extend([exp.result[-2], exp.result[-1]])
    exp.likelihood_estimation()
    results.append(exp.result[-1])
    return results,  exp.result


def main():
    np.random.seed(19971107)
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    plt.style.use('ggplot')
    for i, setting in enumerate(settings):
        fig, ax = plt.subplots(figsize=[8, 6])
        experiment(dim=4, order=setting[0], size_est=5000, sn=setting[1], size_kn=300, ratio=1000, ax=ax)
        ax.legend(loc=1)
        name = '4D, M' + str(setting[0]) + ', SN(' + str(setting[1]) + ')'
        ax.set_title(name)
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    main()
