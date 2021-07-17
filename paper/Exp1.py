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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")


class AdaptiveKDE:
    def __init__(self, centers, weights, bw, fs, a):
        self.centers = centers
        cov = np.cov(centers.T, aweights=weights)
        self.weights = weights / weights.sum()
        self.neff = 1 / np.sum(self.weights ** 2)
        scott = self.neff ** (-1 / (centers.shape[1] + 4))
        self.factor = bw * scott
        self.cov = (self.factor ** 2) * cov
        self.gm = gmean(fs)
        self.h2s = (fs / self.gm) ** (-2 * a)

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for i, loc in enumerate(self.centers):
            density += self.weights[i] * mvnorm.pdf(x=samples, mean=loc, cov=self.h2s[i] * self.cov)

        return density

    def rvs(self, size):
        sizes = size * self.weights
        remain = 1 * (uniform.rvs(size=self.weights.size) <= (sizes % 1))
        sizes = np.int64(sizes) + remain
        sizes[-1] = size - sizes[:-1].sum()
        cum_sizes = np.append(0, np.cumsum(sizes))

        samples = np.zeros([size, self.centers.shape[1]])
        for i, loc in enumerate(self.centers):
            samples[cum_sizes[i]:cum_sizes[i + 1]] = mvnorm.rvs(size=sizes[i], mean=loc, cov=self.h2s[i] * self.cov)

        return samples


class MLE:
    def __init__(self, d, alpha, size_est, show=True):
        self.show = show
        self.Cache = []
        self.result = []

        self.alpha = alpha
        aVar = np.array([alpha * (1 - alpha), 4 * (alpha * (1 - alpha)) ** 2])
        self.disp('Reference for a-var (prob) [direct, optimal]: {}'.format(np.round(aVar, 6)))

        self.target = lambda x: norm
        self.init_proposal = lambda x: t
        self.init_sampler = lambda size: t
        self.size = size_est

        self.centers = None
        self.weights = None
        self.fs = None
        self.labels = None

        self.proportions = None
        self.kdes = None
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
            self.Cache.append(text)

    @staticmethod
    def __cumu(x):
        return x[:, 3:].sum(axis=1)

    @staticmethod
    def __divi(p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, samples, weights, name, asym=True):
        x = self.__cumu(samples)
        self.eVaR = quantile(x, weights, self.alpha)
        self.result.append(self.eVaR)
        if asym:
            w = weights / np.sum(weights)
            aVar = np.sum((w * (1.0 * (x <= self.eVaR) - self.alpha)) ** 2) * x.size
            ESS = 1 / np.sum(w ** 2)
            weighs_fun = weights * (x <= self.eVaR)
            wf = weighs_fun / np.sum(weighs_fun)
            ESSf = 1 / np.sum(wf ** 2)
            self.disp('{} est: {:.4f}; a-var (prob): {:.6f}; ESS: {:.0f}/{}; ESS(f): {:.0f}/{}'
                      .format(name, self.eVaR, aVar, ESS, x.size, ESSf, x.size))
        else:
            self.disp('{} est: {:.4f}'.format(name, self.eVaR))

        if any(weights < 0):
            weights[weights < 0] = 0
            self.eVaR = quantile(x, weights, self.alpha)
            self.disp('(adjusted) {} est: {:.4f}'.format(name, self.eVaR))

    def initial_estimation(self):
        samples = self.init_sampler(self.size)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        self.__estimate(samples, weights, 'IS')

    def resampling(self, size, ratio):
        samples = self.init_sampler(ratio * size)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        if ratio * size > self.size:
            self.__estimate(samples, weights, 'IS({})'.format(ratio * size))

        p = weights * np.abs(1.0 * (self.__cumu(samples) <= self.eVaR) - self.alpha)
        sizes = np.random.multinomial(n=size, pvals=p / p.sum())

        self.centers = samples[sizes != 0]
        self.weights = sizes[sizes != 0]
        self.disp('Resampling rate: {}/{}'.format(self.centers.shape[0], size))
        self.fs = self.target(self.centers) * np.abs(1.0 * (self.__cumu(self.centers) <= self.eVaR) - self.alpha)

    def __coun(self):
        nums = np.array([[self.weights[self.labels == i].sum(), np.sum(self.labels == i)]
                         for i in range(self.labels.max() + 1)]).T
        self.proportions = nums[0] / nums[0].sum()
        self.disp('Clustering: {}/{}'.format(nums[1], nums[0]))

    def __draw(self):
        df = pd.DataFrame(self.centers, columns=['phi0', 'phi1', 'beta'] +
                                                ['y{}'.format(i + 1) for i in range(self.centers.shape[1] - 3)])
        df['type'] = self.labels
        sb.pairplot(df, hue='type')
        plt.show()

    def clustering(self, seed=0, auto=False, num=2):
        if auto:
            scaler = StandardScaler().fit(self.centers, sample_weight=self.weights)
            kmeans = KMeans(n_clusters=num, random_state=seed).fit(scaler.transform(self.centers),
                                                                   sample_weight=self.weights)
            self.labels = kmeans.labels_
        else:
            index = self.__cumu(self.centers) <= self.eVaR
            centers1 = self.centers[index]
            weights1 = self.weights[index]
            centers2 = self.centers[~index]
            weights2 = self.weights[~index]
            scaler1 = StandardScaler().fit(centers1, sample_weight=weights1)
            scaler2 = StandardScaler().fit(centers2, sample_weight=weights2)
            kmeans1 = KMeans(n_clusters=num, random_state=seed).fit(scaler1.transform(centers1), sample_weight=weights1)
            kmeans2 = KMeans(n_clusters=num, random_state=seed).fit(scaler2.transform(centers2), sample_weight=weights2)
            self.labels = np.ones_like(index, dtype=np.int)
            self.labels[index] = kmeans1.labels_
            self.labels[~index] = kmeans2.labels_ + num

        self.__coun()
        if self.show:
            self.__draw()

    def __groups(self):
        return [(self.labels == i) for i in range(self.labels.max() + 1)]

    def adaptive_kde(self, bw, a):
        self.kdes = []
        for i, labels in enumerate(self.__groups()):
            self.kdes.append(AdaptiveKDE(self.centers[labels], self.weights[labels], bw=bw, fs=self.fs[labels], a=a))
            self.disp('KDE {}: {} ({:.4f}, {:.0f})'.format(i + 1, np.round(np.sqrt(np.diag(self.kdes[-1].cov)), 2),
                                                           self.kdes[-1].factor, self.kdes[-1].neff))

    def proposal(self, bw=1.0, adapt=True, rate=0.9):
        a = 1 / self.centers.shape[1] if adapt else 0
        self.adaptive_kde(bw=bw, a=a)
        self.nonpar_proposal = lambda x: np.sum([p * kde.pdf(x) for p, kde in zip(self.proportions, self.kdes)], axis=0)

        def nonpar_sampler(size):
            sizes = np.round(size * self.proportions).astype(np.int64)
            sizes[-1] = size - sizes[:-1].sum()
            return np.vstack([kde.rvs(sz) for kde, sz in zip(self.kdes, sizes)])

        self.nonpar_sampler = nonpar_sampler

        def controls(x):
            out = np.zeros([self.centers.shape[0] - 1, x.shape[0]])
            for j, loc in enumerate(self.centers[1:]):
                label = self.labels[j + 1]
                cov = (self.fs[j + 1] / self.kdes[label].gm) ** (-2 * a) * self.kdes[label].cov
                out[j] = mvnorm.pdf(x=x, mean=loc, cov=cov)

            return np.array(out) - self.nonpar_proposal(x)

        self.controls = controls
        self.mix_proposal = lambda x: (1 - rate) * self.init_proposal(x) + rate * self.nonpar_proposal(x)
        self.mix_sampler = lambda size: np.vstack([self.init_sampler(size - round(rate * size)),
                                                   self.nonpar_sampler(round(rate * size))])

    def nonparametric_estimation(self):
        samples = self.nonpar_sampler(self.size)
        weights = self.__divi(self.target(samples), self.nonpar_proposal(samples))
        self.__estimate(samples, weights, 'NIS')

        self.samples_ = self.mix_sampler(self.size)
        self.target_ = self.target(self.samples_)
        self.proposal_ = self.mix_proposal(self.samples_)
        weights = self.__divi(self.target_, self.proposal_)
        self.__estimate(self.samples_, weights, 'MIS')
        self.controls_ = self.controls(self.samples_)

    def regression_estimation(self):
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
