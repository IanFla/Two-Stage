import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from pandas_datareader import DataReader
import seaborn as sb
import numdifftools as nd
from wquantiles import quantile

from scipy.stats import norm, t
from scipy.stats import multivariate_normal as mvnorm
from scipy.optimize import minimize
from scipy.stats import gmean

from sklearn.linear_model import LinearRegression as Linear
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

data = DataReader('^GSPC', 'yahoo', dt(2010, 9, 29), dt(2011, 7, 14))
returns = pd.DataFrame(100 * np.diff(np.log(data['Adj Close'])), columns=['dlr'])
returns.index = data.index.values[1:data.index.values.shape[0]]
returns = np.array(returns['dlr'])


class GARCH:
    def __init__(self, ys):
        self.h0 = np.std(ys)
        self.y0 = ys[0]
        self.y1toT = ys[1:]
        self.T = self.y1toT.size
        self.prior_pars = [-1, 2]
        self.rvs_trunc = None
        self.pdf_trunc = None

    def posterior(self, pars, scaler=57):
        neglogpdfp0toT = 0.5 * ((pars[:, 0] - self.prior_pars[0]) / self.prior_pars[1]) ** 2
        h = np.exp(pars[:, 0]) + pars[:, 1] * self.y0 ** 2 + pars[:, 2] * self.h0
        for i in range(self.T):
            neglogpdfp0toT += 0.5 * (self.y1toT[i] ** 2 / h + np.log(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * self.y1toT[i] ** 2 + pars[:, 2] * h

        return neglogpdfp0toT - scaler

    @staticmethod
    def __test(pars):
        return (pars[:, 1] >= 0) & (pars[:, 2] >= 0) & (pars[:, 1] + pars[:, 2] < 1)

    def laplace(self, inflate=2, df=1, p_acc=0.481842):
        cons = ({'type': 'ineq',
                 'fun': lambda pars: np.array([pars[1], pars[2], 1 - pars[1] - pars[2]]),
                 'jac': lambda x: np.array([[0, 1, 0], [0, 0, 1], [0, -1, -1]])})
        target = lambda pars: self.posterior(pars.reshape([1, -1]))
        mu0 = np.array([0, 0.1, 0.7])
        res = minimize(target, mu0, method='SLSQP', constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-100, 'gtol': 1e-100, 'disp': False})
        mu = res['x']
        Sigma = np.linalg.inv(nd.Hessian(target)(mu))
        Sigma[:, 0] *= inflate
        Sigma[0, :] *= inflate
        rvs_full = lambda size: np.array([t.rvs(size=size, df=df, loc=mu[i],
                                                scale=np.sqrt(Sigma[i, i])) for i in range(3)]).T

        def rvs_trunc(size):
            pars = rvs_full(int(2 * size / p_acc))
            good = self.__test(pars)
            return pars[good][:size]

        def pdf_trunc(pars):
            good = self.__test(pars)
            pdf = np.prod([t.pdf(x=pars[:, i], df=df, loc=mu[i],
                                 scale=np.sqrt(Sigma[i, i])) for i in range(3)], axis=0)
            return good * pdf / p_acc

        self.rvs_trunc = rvs_trunc
        self.pdf_trunc = pdf_trunc

    @staticmethod
    def __supp(h, ub=1e300):
        h[h > ub] = ub
        return h

    def process(self, pars):
        h = np.exp(pars[:, 0]) + pars[:, 1] * self.y0 ** 2 + pars[:, 2] * self.h0
        for i in range(self.T):
            h = np.exp(pars[:, 0]) + pars[:, 1] * self.y1toT[i] ** 2 + pars[:, 2] * h

        return self.__supp(h)

    def predict(self, d, size):
        pars = self.rvs_trunc(size)
        h = self.process(pars)
        ypre = np.zeros([size, d])
        for i in range(d - 1):
            ypre[:, i] = norm.rvs(scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        half = size // 2
        ypre[:half, -1] = norm.rvs(scale=np.sqrt(h[:half]))
        ypre[half:, -1] = norm.rvs(loc=-np.sqrt(h[half:]), scale=np.sqrt(h[half:]))
        return pars, ypre

    def proposal(self, pars, ypre):
        good = self.__test(pars)
        pars = pars[good]
        ypre = ypre[good]
        h = self.process(pars)
        pdfq = self.pdf_trunc(pars)
        for i in range(ypre.shape[1] - 1):
            pdfq *= norm.pdf(x=ypre[:, i], scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        pdfq *= (norm.pdf(x=ypre[:, -1], scale=np.sqrt(h)) +
                 norm.pdf(x=ypre[:, -1], loc=-np.sqrt(h), scale=np.sqrt(h))) / 2

        out = 1.0 * np.zeros_like(good)
        out[good] = pdfq
        return out

    def target(self, pars, ypre):
        good = self.__test(pars)
        pars = pars[good]
        ypre = ypre[good]
        h = self.process(pars)
        pdfp = np.exp(-self.posterior(pars))
        for i in range(ypre.shape[1]):
            pdfp *= norm.pdf(x=ypre[:, i], scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        out = 1.0 * np.zeros_like(good)
        out[good] = pdfp
        return out


garch = GARCH(returns)
garch.laplace(inflate=2, df=1)


class AdaptiveKDE:
    def __init__(self, centers, weights, bw, fs, a):
        self.centers = centers
        cov = np.cov(centers.T, fweights=weights)
        self.weights = weights / weights.sum()
        self.neff = 1 / np.sum(self.weights ** 2)
        scott = self.neff ** (-1 / (centers.shape[1] + 4))
        self.factor = bw * scott
        self.cov = (self.factor ** 2) * cov
        self.gm = gmean(fs)
        self.hs = (fs / self.gm) ** (-2 * a)

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for i, loc in enumerate(self.centers):
            density += self.weights[i] * mvnorm.pdf(x=samples, mean=loc, cov=self.hs[i] * self.cov)

        return density

    def rvs(self, size):
        samples = np.zeros([size, self.centers.shape[1]])
        sizes = np.random.multinomial(n=size, pvals=self.weights)
        cum_sizes = np.append(0, np.cumsum(sizes))
        for i, loc in enumerate(self.centers):
            samples[cum_sizes[i]:cum_sizes[i + 1]] = mvnorm.rvs(size=sizes[i], mean=loc, cov=self.hs[i] * self.cov)

        return samples


class GroupSampling:
    def __init__(self, d, target, proposal, size, ratio):
        self.d = d
        samples = proposal(size=2 * (ratio * size + self.d))[np.random.permutation(
                  np.arange(2 * (ratio * size + self.d)))]
        pdf = target(samples)
        self.target = pdf[pdf != 0][: ratio * size + self.d]
        self.samples = samples[pdf != 0][: ratio * size + self.d]
        self.size = size
        self.ratio = ratio
        self.index = np.arange(self.d)

    def __dist(self, ind, index):
        dists = np.sqrt(((self.samples[ind] - self.samples[index]) ** 2).sum(axis=1))
        sort = np.argsort(dists)
        dists = dists[sort[:self.d]]
        index = index[sort[:self.d]]
        pdf = np.sqrt(gmean(self.target[index]) * self.target[ind])
        return np.prod(dists) * pdf

    def group_resampling(self):
        for i in range(self.size):
            group = self.d + self.ratio * i + np.arange(self.ratio)
            dist = np.array([self.__dist(ind, self.index) for ind in group])
            self.index = np.append(self.index, group[dist == dist.max(initial=0)])

    def get_samples(self):
        return self.samples[self.index]


class MLE:
    def __init__(self, d, alpha, size_est, show=True):
        self.show = show
        self.Cache = []
        self.result = []

        self.alpha = alpha
        aVar = np.array([alpha * (1 - alpha), 4 * (alpha * (1 - alpha)) ** 2])
        self.disp('Reference for a-var (prob) [direct, optimal]: {}'.format(np.round(aVar, 6)))

        self.target = lambda x: garch.target(x[:, :3], x[:, 3:])
        self.init_proposal = lambda x: garch.proposal(x[:, :3], x[:, 3:])
        self.init_sampler = lambda size: np.hstack(garch.predict(d, size))
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

    @staticmethod
    def __onehot(p, threshold):
        if p.max() / p.sum() >= threshold:
            p[p != p.max()] = 0

        return p

    def shuffling(self, threshold=0.9):
        ps = np.array([kde.pdf(self.centers) for kde in self.kdes]).T
        for i, sample in enumerate(self.centers):
            p = self.__onehot(ps[i], threshold=threshold)
            self.labels[i] = np.random.choice(np.arange(self.labels.max() + 1), 1, p=p / p.sum())

        self.__coun()
        if self.show:
            self.__draw()

    def proposal(self, bw=1.0, adapt=True, rate=0.9, time=0, period=10):
        a = 1 / self.centers.shape[1] if adapt else 0
        self.disp('Original KDE:')
        self.adaptive_kde(bw=bw, a=a)
        flag = True if self.show else False
        for i in range(time):
            if flag:
                self.show = True if (i + 1) % period == 0 else False

            self.shuffling()
            self.disp('Shuffled({}) KDE:'.format(i + 1))
            self.adaptive_kde(bw=bw, a=a)

        self.show = flag
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

    def gpsampling(self, size, ratio):
        gptarget = lambda x: self.target(x) * np.abs(1.0 * (self.__cumu(x) <= self.eVaR) - self.alpha)
        gp = GroupSampling(d=self.centers.shape[1], target=gptarget, proposal=self.mix_sampler, size=size, ratio=ratio)
        gp.group_resampling()
        self.centers = gp.get_samples()
        self.weights = np.ones(self.centers.shape[0], dtype=np.int64)
        self.fs = gptarget(self.centers)

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
    np.random.seed(19971107)
    print('---> Start {} {} <---'.format(pars[0], pars[1]))
    mle = MLE(d=pars[0], alpha=pars[1], size_est=100000, show=True)
    mle.disp('Reference for VaR{} (d={}): {}'.format(pars[1], pars[0], Truth[D == pars[0], Alpha == pars[1]]))
    mle.disp('==IS==================================================IS==')
    mle.initial_estimation()
    mle.resampling(size=size, ratio=1000)
    mle.disp('==NIS================================================NIS==')
    mle.clustering(auto=False, num=2)
    mle.proposal(bw=bw, adapt=True, rate=0.9, time=0, period=5)
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
