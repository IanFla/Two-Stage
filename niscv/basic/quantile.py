import numpy as np
from matplotlib import pyplot as plt
from wquantiles import quantile
from particles import resampling as rs
from niscv.basic.kde import KDE
import sklearn.linear_model as lm
from datetime import datetime as dt
import scipy.optimize as opt

import scipy.stats as st
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
        self.reg_proposal = None

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
        size_est = np.round(ratio * size_kn).astype(np.int64)
        samples = self.init_sampler(size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        statistics = self.statistic(samples)
        self.__estimate(weights, statistics, 'IS')

        VaR = self.result[-2]
        self.opt_proposal = lambda x: self.target(x) * np.abs((self.statistic(x) <= VaR) - self.alpha)
        weights_kn = self.__divi(self.opt_proposal(samples), self.init_proposal(samples))
        ESS = 1 / np.sum((weights_kn / weights_kn.sum()) ** 2)
        RSS = weights_kn.sum() / weights_kn.max()
        self.disp('Ratio reference: n0/ESS {:.0f} ~ n0/RSS {:.0f}'.format(size_est / ESS, size_est / RSS))
        self.result.extend([ESS, RSS])

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
        statistics = self.statistic(samples)
        self.__estimate(weights, statistics, 'NIS')

        samples_ = self.mix_sampler(self.size_est)
        self.target_ = self.target(samples_)
        self.statistics_ = self.statistic(samples_)
        self.proposal_ = self.mix_proposal(samples_)
        self.weights_ = self.__divi(self.target_, self.proposal_)
        self.__estimate(self.weights_, self.statistics_, 'MIS')
        self.controls_ = self.controls(samples_)
        self.VaR = self.result[-2]

    def regression_estimation(self):
        X = (self.__divi(self.controls_, self.proposal_)).T
        w = self.weights_
        y = w * (self.statistics_ <= self.VaR)
        yw = y - self.alpha * w

        reg1 = lm.LinearRegression().fit(X, y)
        reg2 = lm.LinearRegression().fit(X, w)
        reg3 = lm.LinearRegression().fit(X, yw)
        self.disp('Regression R2: {:.4f} ({:.4f} / {:.4f})'
                  .format(reg3.score(X, yw), reg1.score(X, y), reg2.score(X, w)))
        self.result.extend([reg3.score(X, yw), reg1.score(X, y), reg2.score(X, w)])
        self.reg_proposal = lambda x: reg3.coef_.dot(self.controls(x))

        zeta = np.linalg.solve(np.cov(X.T, bias=True), X.mean(axis=0))
        weights = self.weights_ * (1 - (X - X.mean(axis=0)).dot(zeta))
        self.disp('Reg weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                  .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
        self.__estimate(weights, self.statistics_, 'RIS', asym=False)
        avar = np.mean(((yw - X.dot(reg3.coef_)) / np.mean(w - X.dot(reg2.coef_))) ** 2)
        self.disp('RIS a-var: {:.4f}'.format(avar))
        self.result.append(avar)

    def likelihood_estimation(self, optimize=True, NR=True):
        target = lambda zeta: -np.mean(np.log(self.proposal_ + zeta.dot(self.controls_)))
        gradient = lambda zeta: -np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_))
                                           ** 2).dot(self.controls_.T) / self.controls_.shape[1]
        zeta0 = np.zeros(self.controls_.shape[0])
        if optimize:
            begin = dt.now()
            if NR:
                res = opt.root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta0, method='lm', jac=True)
            else:
                cons = ({'type': 'ineq', 'fun': lambda zeta: self.proposal_ + zeta.dot(self.controls_),
                         'jac': lambda zeta: self.controls_.T})
                res = opt.minimize(target, zeta0, method='SLSQP', jac=gradient, constraints=cons,
                                   options={'ftol': 1e-8, 'maxiter': 1000})

            end = dt.now()
            self.disp('')
            self.disp('Optimization results (spent {} seconds):'.format((end - begin).seconds))
            if res['success']:
                zeta1 = res['x']
                grad = gradient(zeta1)
                self.disp('Optimal: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})'
                          .format(target(zeta1), grad.min(), grad.mean(), grad.max(), grad.std()))
                weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
                self.disp('MLE weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                          .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
                self.__estimate(weights, self.statistics_, 'MLE', asym=False)
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
            ax.plot(grid_x, np.abs(self.reg_proposal(grid_X)))
            ax.legend(['optimal proposal', 'regression proposal'])
        else:
            print('name err! ')

        ax.set_title('{}-D {} estimation ({}th slicing)'.format(self.dim, name, d + 1))
        plt.show()


def experiment(dim, alpha, size_est, show, size_kn, ratio):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    statistic = lambda x: x[:, 0]
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    qtl = Quantile(dim, target, statistic, alpha, init_proposal, size_est, show=show)
    qtl.initial_estimation(size_kn, ratio)
    results.extend([qtl.result[-5], qtl.result[-4]])
    if qtl.show:
        qtl.draw(grid_x, name='initial')

    qtl.density_estimation(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    qtl.nonparametric_estimation()
    results.extend([qtl.result[-4], qtl.result[-3], qtl.result[-2], qtl.result[-1]])
    if qtl.show:
        qtl.draw(grid_x, name='nonparametric')

    qtl.regression_estimation()
    results.extend([qtl.result[-2], qtl.result[-1]])
    if qtl.show:
        qtl.draw(grid_x, name='regression')

    qtl.likelihood_estimation(optimize=True, NR=True)
    results.extend([qtl.result[-1], results[-1]])
    return results


def main():
    np.random.seed(3033079628)
    results = []
    for i in range(100):
        print(i + 1)
        result = experiment(dim=4, alpha=0.05, size_est=25000, show=False, size_kn=500, ratio=20)
        results.append(result)

    return np.array(results)


if __name__ == '__main__':
    truth = st.norm.ppf(0.05)
    pdf = st.norm.pdf(truth)
    R = main()

    aVar = R[:, 1::2] / (pdf ** 2)
    mean_aVar = aVar.mean(axis=0)
    std_aVar = aVar.std(axis=0)
    print('a-var(l):', np.round(mean_aVar - 1.96 * std_aVar, 4))
    print('a-var(m):', np.round(mean_aVar, 4))
    print('a-var(r):', np.round(mean_aVar + 1.96 * std_aVar, 4))

    MSE = np.mean((R[:, ::2] - truth) ** 2, axis=0)
    print('nMSE:', np.round(np.append(10000 * MSE[0], 25000 * MSE[1:]), 4))

    aErr = np.sqrt(np.hstack([aVar[:, 0].reshape([-1, 1]) / 10000, aVar[:, 1:] / 25000]))
    Flag = (truth >= R[:, ::2] - 1.96 * aErr) & (truth <= R[:, ::2] + 1.96 * aErr)
    print('C.I.:', Flag.mean(axis=0))
