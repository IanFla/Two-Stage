import numpy as np
import scipy.stats as st
from niscv.clustering.probability import Probability
from matplotlib import pyplot as plt


def experiment(dim, b, size_est, size_kn, ratio, ax):
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    indicator = lambda x: 1 * (x[:, 0] >= b)
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    grid_X = np.zeros([grid_x.size, dim])
    grid_X[:, 0] = grid_x

    pro = Probability(dim, target, indicator, init_proposal, size_est, show=False)
    pro.initial_estimation(size_kn, ratio, resample=True)
    opt = pro.opt_proposal(grid_X)
    init = pro.init_proposal(grid_X)
    ax.plot(grid_x, opt, c='k', label='optimal proposal')
    ax.plot(grid_x, opt.max() * init / init.max() / 10, c='b', label='initial proposal')
    pro.density_estimation(bw=1.0, mode=0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    pro.nonparametric_estimation()
    nonpar = pro.nonpar_proposal(grid_X)
    mix = pro.mix_proposal(grid_X)
    ax.plot(grid_x, opt.max() * nonpar / nonpar.max(), c='y', label='nonparametric proposal')
    ax.plot(grid_x, opt.max() * mix / mix.max(), c='c', label='mixture proposal')
    pro.regression_estimation()
    reg = (pro.reg1.coef_ - pro.mu * pro.reg2.coef_).dot(pro.controls(grid_X))
    ax.plot(grid_x, np.abs(reg), c='r', label='regression proposal')


def main():
    np.random.seed(19971107)
    plt.style.use('ggplot')
    for i in range(4):
        fig, ax = plt.subplots(figsize=[8, 6])
        experiment(dim=5, b=i, size_est=10000, size_kn=500, ratio=1000, ax=ax)
        ax.legend(loc=1)
        ax.set_title('b(' + str(i) + ')')
        fig.tight_layout()
        fig.show()


if __name__ == '__main__':
    main()
