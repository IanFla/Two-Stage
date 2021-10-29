import numpy as np
from matplotlib import pyplot as plt
from wquantiles import quantile
from particles import resampling as rs
from kde import KDE
from datetime import datetime as dt
import scipy.optimize as opt

# import scipy.stats as st
# import pickle
# import multiprocessing
import warnings
warnings.filterwarnings("ignore")


class Quantile:
    def __init__(self, dim, target, statistic, alpha, init_proposal, size_est, show=True):
        self.dim = dim
        self.show = show
        self.cache = []
        self.result = []

        self.target = target
        self.statistic = statistic
        self.alpha = alpha
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

        self.target_ = None
        self.statistics_ = None
        self.proposal_ = None
        self.weights_ = None

        self.controls_ = None
        self.VaR = None

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights, statistics, name, asym=True):
        VaR = quantile(statistics, weights, self.alpha)
        self.result.append(VaR)
        if asym:
            avar = np.mean((weights * ((statistics <= VaR) - self.alpha) / np.mean(weights)) ** 2)
            self.result.append(avar)
            self.disp('{} est: {:.4f}; a-var: {:.4f}'.format(name, VaR, avar))
        else:
            self.disp('{} est: {:.4f}'.format(name, VaR))

    def initial_estimation(self, size_kn, ratio):
        size_est = ratio * size_kn
        samples = self.init_sampler(size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'IS')

        mu = self.result[-2]
        self.opt_proposal = (lambda x: self.target(x) * np.abs(self.fun(x) - mu)) \
            if self.sn else (lambda x: self.target(x) * np.abs(self.fun(x)))

        weights_kn = self.__divi(self.opt_proposal(samples), self.init_proposal(samples))
        ESS = 1 / np.sum((weights_kn / weights_kn.sum()) ** 2)
        RSS = weights_kn.sum() / weights_kn.max()
        self.disp('Ratio reference: n0/ESS {:.0f} ~ n0/RSS {:.0f}'.format(size_est / ESS, size_est / RSS))

        index, sizes = np.unique(rs.stratified(weights_kn / weights_kn.sum(), M=size_kn), return_counts=True)
        self.centers = samples[index]
        self.weights_kn = sizes
        self.disp('Resampling rate: {}/{}'.format(self.weights_kn.size, size_kn))
        self.result.append(self.weights_kn.size)

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

    def regression_estimation(self):
        self.controls_ = self.controls(self.samples_)
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * self.fun(self.samples_)

        if self.sn:
            self.reg1 = lm.LinearRegression().fit(X, y)
            self.reg2 = lm.LinearRegression().fit(X, w)
            self.disp('Regression R2: {:.4f} / {:.4f}'.format(self.reg1.score(X, y), self.reg2.score(X, w)))
            self.result.extend([self.reg1.score(X, y), self.reg2.score(X, w)])
        else:
            self.reg = lm.LinearRegression().fit(X, y)
            self.disp('Regression R2: {:.4f}'.format(self.reg.score(X, y)))
            self.result.append(self.reg.score(X, y))

        self.__estimate(self.weights_, self.fun_, 'RIS', mode='regression')

    def likelihood_estimation(self, optimize=True, NR=True):
        target = lambda zeta: -np.mean(np.log(self.proposal_ + zeta.dot(self.controls_)))
        gradient = lambda zeta: -np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_))
                                           ** 2).dot(self.controls_.T) / self.controls_.shape[1]
        zeta0 = np.zeros(self.controls_.shape[0])
        grad0 = gradient(zeta0)
        self.disp('')
        self.disp('MLE reference:')
        self.disp('Origin: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})'
                  .format(target(zeta0), grad0.min(), grad0.mean(), grad0.max(), grad0.std()))

        self.disp('')
        self.disp('Theoretical results:')
        X = (self.__divi(self.controls_, self.proposal_)).T
        zeta1 = np.linalg.solve(np.cov(X.T, bias=True), X.mean(axis=0))
        self.disp('MLE(The) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})'
                  .format(zeta1.min(), zeta1.mean(), zeta1.max(), zeta1.std(), np.sqrt(np.sum(zeta1 ** 2))))
        grad1 = gradient(zeta1)
        self.disp('Theory: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})'
                  .format(target(zeta1), grad1.min(), grad1.mean(), grad1.max(), grad1.std()))
        weights = self.weights_ * (1 - (X - X.mean(axis=0)).dot(zeta1))
        self.disp('Reg weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                  .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
        self.__estimate(weights, self.fun_, 'RIS(The)', mode='likelihood')
        weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
        self.disp('MLE weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                  .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
        self.__estimate(weights, self.fun_, 'MLE(The)', mode='likelihood')

        if optimize:
            zeta2 = zeta0 if np.isnan(target(zeta1)) else zeta1
            begin = dt.now()
            if NR:
                res = opt.root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta2, method='lm', jac=True)
            else:
                cons = ({'type': 'ineq', 'fun': lambda zeta: self.proposal_ + zeta.dot(self.controls_),
                         'jac': lambda zeta: self.controls_.T})
                res = opt.minimize(target, zeta2, method='SLSQP', jac=gradient, constraints=cons,
                                   options={'ftol': 1e-8, 'maxiter': 1000})

            end = dt.now()
            self.disp('')
            self.disp('Optimization results (spent {} seconds):'.format((end - begin).seconds))
            if res['success']:
                zeta2 = res['x']
                self.disp('MLE(Opt) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})'
                          .format(zeta2.min(), zeta2.mean(), zeta2.max(), zeta2.std(), np.sqrt(np.sum(zeta2 ** 2))))
                self.disp('Dist(zeta(Opt),zeta(The))={:.4f}'.format(np.sqrt(np.sum((zeta2 - zeta1) ** 2))))
                grad2 = gradient(zeta2)
                self.disp('Optimal: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})'
                          .format(target(zeta2), grad2.min(), grad2.mean(), grad2.max(), grad2.std()))
                weights = self.__divi(self.target_, self.proposal_ + zeta2.dot(self.controls_))
                self.disp('MLE weights (Opt): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                          .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
                self.__estimate(weights, self.fun_, 'MLE(Opt)', mode='likelihood')
            else:
                self.disp('MLE fail')

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
        elif name == 'regression':
            reg_proposal = (self.reg1.coef_ - self.mu * self.reg2.coef_).dot(self.controls(grid_X)) \
                if self.sn else self.reg.coef_.dot(self.controls(grid_X)) + self.mu * self.mix_proposal(grid_X)
            ax.plot(grid_x, np.abs(reg_proposal))
            ax.legend(['optimal proposal', 'regression proposal'])
        else:
            print('name err! ')

        ax.set_title('{}-D {} estimation ({}th slicing)'.format(self.dim, name, d + 1))
        plt.show()


def main():
    pass


if __name__ == '__main__':
    pass
