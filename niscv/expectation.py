import numpy as np
from matplotlib import pyplot as plt
from particles import resampling as rs
from kde import KDE
# import sklearn.linear_model as lmd
# import scipy.optimize as opt

import scipy.stats as st
# from datetime import datetime as dt
# import pickle
# import multiprocessing
# import warnings
# warnings.filterwarnings("ignore")


class Expectation:
    def __init__(self, dim, target, fun, init_proposal, size_est, sn=False, show=True):
        self.dim = dim
        self.sn = sn
        self.show = show
        self.cache = []
        self.result = []

        self.target = target
        self.fun = fun
        self.init_proposal = init_proposal.pdf
        self.init_sampler = init_proposal.rvs
        self.size_est = size_est

        self.opt_proposal = None
        self.centers = None
        self.weights_kn = None

        self.kde = None
        self.nonpar_proposal = None
        self.nonpar_sampler = None
        self.mix_proposal = None
        self.mix_sampler = None
        self.controls = None

        self.samples_ = None
        self.target_ = None
        self.fun_ = None
        self.proposal_ = None
        self.weights_ = None

        self.controls_ = None
        self.regO = None
        self.regR = None
        self.regL = None

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights, funs, name, asymp=True):
        mu = np.sum(weights * funs) / np.sum(weights) if self.sn else np.mean(weights * funs)
        self.result.append(mu)
        if asymp:
            avar = np.mean((weights * (funs - mu) / np.mean(weights)) ** 2) if self.sn else np.var(weights * funs)
            self.result.append(avar)
            aerr = np.sqrt(avar / weights.size)
            self.disp('{} est: {:.4f}; a-var: {:.4f}; 95% C.I.: [{:.4f}, {:.4f}]'
                      .format(name, mu, avar, mu - 1.96 * aerr, mu + 1.96 * aerr))
        else:
            self.disp('{} est: {:.4f}'.format(name, mu))

    def initial_estimation(self, size_kn, ratio, resample=True):
        size_est = ratio * size_kn
        samples = self.init_sampler(size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'IS')

        mu = self.result[-2]
        self.opt_proposal = (lambda x: self.target(x) * np.abs(self.fun(x) - mu)) \
            if self.sn else (lambda x: self.target(x) * np.abs(self.fun(x)))

        if resample:
            weights_kn = self.__divi(self.opt_proposal(samples), self.init_proposal(samples))
            ESS = 1 / np.sum((weights_kn / weights_kn.sum()) ** 2)
            RSS = weights_kn.sum() / weights_kn.max()
            self.disp('Ratio reference: n0/ESS {:.0f} ~ n0/RSS {:.0f}'.format(size_est / ESS, size_est / RSS))

            index, sizes = np.unique(rs.stratified(weights_kn / weights_kn.sum(), M=size_kn), return_counts=True)
            self.centers = samples[index]
            self.weights_kn = sizes
            self.disp('Resampling rate: {}/{}'.format(self.weights_kn.size, size_kn))
            self.result.append(self.weights_kn.size)
        else:
            self.centers = self.init_sampler(size_kn)
            self.weights_kn = self.__divi(self.opt_proposal(self.centers), self.init_proposal(self.centers))

    def density_estimation(self, bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1):
        self.kde = KDE(self.centers, self.weights_kn, bw=bw, factor=factor, local=local, gamma=gamma, df=df)
        bdwths = np.sqrt(np.diag(self.kde.covs.mean(axis=0) if local else self.kde.covs))
        self.disp('KDE: (bdwth: {:.4f} ({:.4f}), factor {:.4f})'
                  .format(bdwths[0], np.mean(bdwths[1:]), self.kde.factor))
        self.result.extend([bdwths[0], np.mean(bdwths[1:])])

        self.nonpar_proposal = self.kde.pdf
        self.nonpar_sampler = self.kde.rvs
        self.mix_proposal = lambda x: alpha0 * self.init_proposal(x) + (1 - alpha0) * self.nonpar_proposal(x)
        self.mix_sampler = lambda size: np.vstack([self.init_sampler(round(alpha0 * size)),
                                                   self.nonpar_sampler(size - round(alpha0 * size))])

        def controls(x):
            out = np.zeros([self.centers.shape[0], x.shape[0]])
            for j, center in enumerate(self.centers):
                cov = self.kde.covs[j] if local else self.kde.covs
                out[j] = self.kde.kernel_pdf(x=x, m=center, v=cov)

            return out - self.mix_proposal(x)

        self.controls = controls

    def nonparametric_estimation(self):
        samples = self.nonpar_sampler(self.size_est)
        weights = self.__divi(self.target(samples), self.nonpar_proposal(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'NIS')

        self.samples_ = self.mix_sampler(self.size_est)
        self.target_ = self.target(self.samples_)
        self.fun_ = self.fun(self.samples_)
        self.proposal_ = self.mix_proposal(self.samples_)
        self.weights_ = self.__divi(self.target_, self.proposal_)
        self.__estimate(self.weights_, self.fun_, 'MIS')

    def draw(self, grid_x, name, d=0):
        grid_X = np.zeros([grid_x.size, self.dim])
        grid_X[:, d] = grid_x
        opt_proposal = self.opt_proposal(grid_X)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grid_x, opt_proposal)
        if name == 'initial':
            init_proposal = self.init_proposal(grid_X)
            ax.plot(grid_x, opt_proposal.max() * init_proposal / init_proposal.max())
            ax.legend(['optimal proposal', 'initial proposal'])
        elif name == 'nonparametric':
            nonpar_proposal = self.nonpar_proposal(grid_X)
            mix_proposal = self.mix_proposal(grid_X)
            ax.plot(grid_x, opt_proposal.max() * nonpar_proposal / nonpar_proposal.max())
            ax.plot(grid_x, opt_proposal.max() * mix_proposal / mix_proposal.max())
            ax.legend(['optimal proposal', 'nonparametric proposal', 'mixture proposal'])
        else:
            print('name err! ')

        ax.set_title('{}-D {} estimation ({}th slicing)'.format(self.dim, name, d + 1))
        plt.show()


def experiment(dim, size_est, sn, size_kn, ratio):
    mean = np.zeros(dim)
    target = st.multivariate_normal(mean=mean).pdf
    fun = lambda x: x[:, 0]
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    exp = Expectation(dim, target, fun, init_proposal, size_est, sn=sn, show=True)
    exp.initial_estimation(size_kn, ratio, resample=True)
    exp.draw(grid_x, name='initial')
    exp.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    exp.nonparametric_estimation()
    exp.draw(grid_x, name='nonparametric')


def main():
    experiment(dim=2, size_est=10000, sn=True, size_kn=1000, ratio=20)


if __name__ == '__main__':
    main()
