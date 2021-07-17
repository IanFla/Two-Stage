import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
import seaborn as sb
import numdifftools as nd
from wquantiles import quantile

from scipy.stats import norm, t, uniform
from scipy.stats import multivariate_normal as mvnorm
from scipy.optimize import minimize
from scipy.stats import gmean

from sklearn.linear_model import LinearRegression as Linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


class KDE:
    def __init__(self, centers, weights, bw, local, gamma=None, ps=None, a=None):
        self.centers = centers
        self.weights = weights / weights.sum()
        self.size, self.d = centers.shape
        self.local = local
        if self.local:
            self.gamma = gamma
            self.neff = self.gamma * self.size
            scaler = StandardScaler().fit(self.centers, sample_weight=self.weights)
            standard_centers = scaler.transform(self.centers)
            covs = []
            for center in standard_centers:
                index = self.dist(center, standard_centers)
                cov = np.cov(self.centers[index].T, aweights=weights[index])
                covs.append(cov)

        else:
            self.neff = 1 / np.sum(self.weights ** 2)
            covs = np.cov(centers.T, aweights=weights)
            self.gm = gmean(ps)
            self.lambda2s = (ps / self.gm) ** (-2 * a)

        scott = self.neff ** (-1 / (self.d + 4))
        self.factor = bw * scott
        self.covs = (self.factor ** 2) * np.array(covs)

    def dist(self, x, X):
        distances = np.sum((x - X) ** 2, axis=1)
        return np.argsort(distances)[:np.around(self.gamma * self.size).astype(np.int64)]

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.lambda2s[j] * self.covs
            density += self.weights[j] * mvnorm.pdf(x=samples, mean=center, cov=cov)

        return density

    def rvs(self, size):
        sizes = size * self.weights
        remain = 1 * (uniform.rvs(size=self.weights.size) <= (sizes % 1))
        sizes = np.int64(sizes) + remain
        sizes[-1] = size - sizes[:-1].sum()
        cum_sizes = np.append(0, np.cumsum(sizes))

        samples = np.zeros([size, self.centers.shape[1]])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.lambda2s[j] * self.covs
            samples[cum_sizes[j]:cum_sizes[j + 1]] = mvnorm.rvs(size=sizes[j], mean=center, cov=cov)

        return samples


class MLE:
    def __init__(self, target, init_proposal, size_est, show=True):
        self.show = show
        self.cache = []
        self.result = []

        self.target = target.pdf
        self.init_proposal = init_proposal.pdf
        self.init_sampler = init_proposal.rvs
        self.size_est = size_est
        self.Z = None

        self.centers = None
        self.weights = None
        self.ps = None

        self.kde = None
        self.nonpar_proposal = None
        self.nonpar_sampler = None
        self.controls = None
        self.mix_proposal = None
        self.mix_sampler = None

        self.samples_ = None
        self.target_ = None
        self.proposal_ = None
        self.controls_ = None

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.cache.append(text)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, weights, name, asym=True):
        Z = np.mean(weights)
        Err = np.abs(Z - 1)
        self.result.append(Z)
        if asym:
            aVar = np.var(weights)
            aErr = np.sqrt(aVar / weights.size)
            ESS = 1 / np.sum((weights / np.sum(weights)) ** 2)
            self.disp('{} est: {:.4f}; err: {:.4f}; a-var: {:.4f}; a-err: {:.4f}; ESS: {:.0f}/{}'
                      .format(name, Z, Err, aVar, aErr, ESS, weights.size))
        else:
            self.disp('{} est: {:.4f}; err: {:.4f}'.format(name, Z, Err))

    def initial_estimation(self):
        samples = self.init_sampler(self.size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        self.__estimate(weights, 'IS')

    def resampling(self, size, ratio, resample=True):
        if resample:
            samples = self.init_sampler(ratio * size)
            weights = self.__divi(self.target(samples), self.init_proposal(samples))
            if ratio * size > self.size_est:
                self.__estimate(weights, 'IS({})'.format(ratio * size))

            sizes = np.random.multinomial(n=size, pvals=weights / weights.sum())

            self.centers = samples[sizes != 0]
            self.weights = sizes[sizes != 0]
            self.disp('Resampling rate: {}/{}'.format(self.weights.size, size))
        else:
            self.centers = self.init_sampler(size)
            self.weights = self.__divi(self.target(self.centers), self.init_proposal(self.centers))

        self.ps = self.target(self.centers)
        self.ps /= gmean(self.ps)

    def proposal(self, bw=1.0, local=False, gamma=0.1, a=0.5, rate=0.9):
        self.kde = KDE(self.centers, self.weights, bw=bw, local=local, gamma=gamma, ps=self.ps, a=a)
        self.disp('KDE: (factor {:.4f}, ESS {:.0f})'.format(self.kde.factor, self.kde.neff))
        self.nonpar_proposal = self.kde.pdf
        self.nonpar_sampler = self.kde.rvs
        self.mix_proposal = lambda x: (1 - rate) * self.init_proposal(x) + rate * self.nonpar_proposal(x)
        self.mix_sampler = lambda size: np.vstack([self.init_sampler(size - round(rate * size)),
                                                   self.nonpar_sampler(round(rate * size))])

        def controls(x):
            out = np.zeros([self.centers.shape[0], x.shape[0]])
            for j, center in enumerate(self.centers):
                cov = self.kde.covs[j] if local else self.kde.lambda2s[j] * self.kde.covs
                out[j] = mvnorm.pdf(x=x, mean=center, cov=cov)

            return np.array(out) - self.nonpar_proposal(x)

        self.controls = controls

    def nonparametric_estimation(self):
        samples = self.nonpar_sampler(self.size_est)
        weights = self.__divi(self.target(samples), self.nonpar_proposal(samples))
        self.__estimate(weights, 'NIS')

        self.samples_ = self.mix_sampler(self.size_est)
        self.target_ = self.target(self.samples_)
        self.proposal_ = self.mix_proposal(self.samples_)
        weights = self.__divi(self.target_, self.proposal_)
        self.__estimate(weights, 'MIS')

    def regression_estimation(self):
        self.controls_ = self.controls(self.samples_)
        X = (self.__divi(self.controls_, self.proposal_)).T
        tmp = X / np.linalg.norm(X, axis=0)
        lbd = np.linalg.eigvals(tmp.T.dot(tmp))
        tau = np.sqrt(lbd.max(initial=0) / lbd)
        self.disp('Condition index: (min {:.4f}, median {:.4f}, mean {:.4f}, max {:.4f}, [>30] {}/{})'
                  .format(tau.min(), np.median(tau), tau.mean(), tau.max(), np.sum(tau > 30), tau.size))

        y2 = self.__divi(self.target_, self.proposal_)
        y1 = y2 * (self.__cumu(self.samples_) <= self.eVaR)
        y3 = y1 - self.alpha * y2
        reg1 = Linear().fit(X, y1)
        reg2 = Linear().fit(X, y2)
        reg3 = Linear().fit(X, y3)
        self.disp('Tail R2: {:.4f}; Body R2: {:.4f}; Overall R2: {:.4f}'
                  .format(reg1.score(X, y1), reg2.score(X, y2), reg3.score(X, y3)))

        W2 = y2 - X.dot(reg2.coef_)
        W3 = y3 - X.dot(reg3.coef_)
        aVar = W2.size * np.sum(W3 ** 2) / (np.sum(W2)) ** 2
        self.disp('RIS a-var: {:.6f}'.format(aVar))

        XX = X - X.mean(axis=0)
        zeta = np.linalg.solve(XX.T.dot(XX), X.sum(axis=0))
        weights = self.__divi(self.target_, self.proposal_) * (1 - XX.dot(zeta))
        self.disp('reg weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                  .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
        self.__estimate(self.samples_, weights, 'RIS', asym=False)


D = np.array([1, 2, 5])
Alpha = np.array([0.05, 0.01])
Truth = np.array([[-1.333, -1.895], [-1.886, -2.771], [-2.996, -4.424]])
params = [[1500, 1.1], [2000, 1.3], [3000, 1.4]]


def experiment(pars, size, bw):
    np.random.seed(53465496)
    print('---> Start {} {} <---'.format(pars[0], pars[1]))
    mle = MLE(d=pars[0], alpha=pars[1], size_est=100000, show=True)
    mle.disp('Reference for VaR{} (d={}): {}'.format(pars[1], pars[0], Truth[D == pars[0], Alpha == pars[1]]))
    mle.disp('==IS==================================================IS==')
    mle.initial_estimation()
    mle.resampling(size=size, ratio=1000)
    mle.disp('==NIS================================================NIS==')
    mle.clustering(auto=False, num=2)
    mle.proposal(bw=bw, adapt=True, rate=0.9)
    mle.nonparametric_estimation()
    mle.disp('==RIS================================================RIS==')
    mle.regression_estimation()
    print('---> End {} {} <---'.format(pars[0], pars[1]))
    return mle.Cache


def main():
    begin = dt.now()
    Cache = []
    for i, d in enumerate(D):
        for alpha in Alpha:
            Cache.append(experiment((d, alpha), size=params[i][0], bw=params[i][1]))

    end = dt.now()
    print((end - begin).seconds)
    return Cache


if __name__ == '__main__':
    result = main()
