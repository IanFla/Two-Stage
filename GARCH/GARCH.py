import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from pandas_datareader import DataReader as DR
import seaborn as sb
import numdifftools as nd
from wquantiles import quantile
import statsmodels.api as sm
import multiprocessing
import pickle

from scipy.stats import norm, t, truncnorm, multinomial
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import multivariate_t as mvt
from scipy.spatial import Delaunay as TRI
from scipy.interpolate import LinearNDInterpolator as ITP
from scipy.optimize import minimize, root
from scipy.optimize import NonlinearConstraint as NonlinCons
from scipy.stats import gaussian_kde as sciKDE
from scipy.stats import gmean

from sklearn.linear_model import LinearRegression as Linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KernelDensity as sklKDE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

data = DR('^GSPC', 'yahoo', dt(2010, 9, 29), dt(2011, 7, 14))
returns = pd.DataFrame(100 * np.diff(np.log(data['Adj Close'])), columns=['dlr'])
returns.index = data.index.values[1:data.index.values.shape[0]]
returns = np.array(returns['dlr'])


class GARCH:
    def __init__(self, returns):
        self.h0 = np.std(returns)
        self.y0 = returns[0]
        self.YT = returns[1:]
        self.T = self.YT.size
        self.prior_pars = [-1, 2]

    def posterior(self, pars, Z=57):
        neglogpdfpT = 0.5 * ((pars[:, 0] - self.prior_pars[0]) / self.prior_pars[1]) ** 2
        H = np.exp(pars[:, 0]) + pars[:, 1] * self.y0 ** 2 + pars[:, 2] * self.h0
        for i in range(self.T):
            neglogpdfpT += 0.5 * (self.YT[i] ** 2 / H + np.log(H))
            H = np.exp(pars[:, 0]) + pars[:, 1] * self.YT[i] ** 2 + pars[:, 2] * H

        return neglogpdfpT - Z

    def __test(self, pars):
        return (pars[:, 1] >= 0) & (pars[:, 2] >= 0) & (pars[:, 1] + pars[:, 2] < 1)

    def laplace(self, inflate=2, df=1, p_acc=0.481842):
        cons = ({'type': 'ineq', \
                 'fun': lambda pars: np.array([pars[1], pars[2], 1 - pars[1] - pars[2]]), \
                 'jac': lambda x: np.array([[0, 1, 0], [0, 0, 1], [0, -1, -1]])})
        target = lambda pars: self.posterior(pars.reshape([1, -1]))
        res = minimize(target, [0, 0.1, 0.7], method='SLSQP', constraints=cons, \
                       options={'maxiter': 1000, 'ftol': 1e-100, 'gtol': 1e-100, 'disp': False})
        mu = res['x']
        Sigma = np.linalg.inv(nd.Hessian(target)(mu))
        Sigma[:, 0] *= inflate
        Sigma[0, :] *= inflate
        pars_full = lambda size: np.array([t.rvs(size=size, df=df, loc=mu[i], \
                                                 scale=np.sqrt(Sigma[i, i])) for i in range(3)]).T

        def pars_trunc(size):
            pars = pars_full(int(2 * size / p_acc))
            good = self.__test(pars)
            return pars[good][:size]

        def lplc_trunc(pars):
            good = self.__test(pars)
            pdf = np.prod([t.pdf(x=pars[:, i], df=df, loc=mu[i], \
                                 scale=np.sqrt(Sigma[i, i])) for i in range(3)], axis=0)
            return good * pdf / p_acc

        self.pars_trunc = pars_trunc
        self.lplc_trunc = lplc_trunc

    def __supp(self, H, ub=1e300):
        H[H > ub] = ub
        return H

    def process(self, pars):
        H_T1 = np.exp(pars[:, 0]) + pars[:, 1] * self.y0 ** 2 + pars[:, 2] * self.h0
        for i in range(self.T):
            H_T1 = np.exp(pars[:, 0]) + pars[:, 1] * self.YT[i] ** 2 + pars[:, 2] * H_T1

        H_T1 = self.__supp(H_T1)
        return H_T1

    def predict(self, d, size):
        pars = self.pars_trunc(size)
        H_Td = self.process(pars)
        Yd = np.zeros([size, d])
        for i in range(d - 1):
            Yd[:, i] = norm.rvs(scale=np.sqrt(H_Td))
            H_Td = np.exp(pars[:, 0]) + pars[:, 1] * Yd[:, i] ** 2 + pars[:, 2] * H_Td
            H_Td = self.__supp(H_Td)

        half = size // 2
        Yd[:half, -1] = norm.rvs(scale=np.sqrt(H_Td[:half]))
        Yd[half:, -1] = norm.rvs(loc=-np.sqrt(H_Td[half:]), scale=np.sqrt(H_Td[half:]))
        return pars, Yd

    def proposal(self, pars, Yd):
        good = self.__test(pars)
        pars = pars[good]
        Yd = Yd[good]
        H_Td = self.process(pars)
        pdfq = self.lplc_trunc(pars)
        for i in range(Yd.shape[1] - 1):
            pdfq *= norm.pdf(x=Yd[:, i], scale=np.sqrt(H_Td))
            H_Td = np.exp(pars[:, 0]) + pars[:, 1] * Yd[:, i] ** 2 + pars[:, 2] * H_Td
            H_Td = self.__supp(H_Td)

        pdfq *= (norm.pdf(x=Yd[:, -1], scale=np.sqrt(H_Td)) + \
                 norm.pdf(x=Yd[:, -1], loc=-np.sqrt(H_Td), scale=np.sqrt(H_Td))) / 2

        tmp = 1.0 * np.zeros_like(good)
        tmp[good] = pdfq
        return tmp

    def target(self, pars, Yd):
        good = self.__test(pars)
        pars = pars[good]
        Yd = Yd[good]
        H_Td = self.process(pars)
        pdfp = np.exp(-self.posterior(pars))
        for i in range(Yd.shape[1]):
            pdfp *= norm.pdf(x=Yd[:, i], scale=np.sqrt(H_Td))
            H_Td = np.exp(pars[:, 0]) + pars[:, 1] * Yd[:, i] ** 2 + pars[:, 2] * H_Td
            H_Td = self.__supp(H_Td)

        tmp = 1.0 * np.zeros_like(good)
        tmp[good] = pdfp
        return tmp


garch = GARCH(returns)
garch.laplace(inflate=2, df=1)


class AKDE:
    def __init__(self, rS, bw, F, a):
        self.rS = rS
        self.K = rS.shape[0]
        self.f = bw * sciKDE(rS.T, bw_method='silverman').factor
        self.cov = np.cov(rS.T)
        self.gm = gmean(F)
        self.H2 = (F / self.gm) ** (-2 * a)

    def pdf(self, S):
        res = np.zeros(S.shape[0])
        for i, loc in enumerate(self.rS):
            res += mvnorm.pdf(x=S, mean=loc, cov=self.f * self.H2[i] * self.cov)

        return res / self.K

    def rvs(self, size):
        res = np.zeros([size, self.rS.shape[1]])
        sizes = multinomial.rvs(n=size, p=np.ones(self.K) / self.K)
        cumsizes = np.append(0, np.cumsum(sizes))
        for i, loc in enumerate(self.rS):
            res[cumsizes[i]:cumsizes[i + 1], :] = mvnorm.rvs(size=sizes[i], mean=loc,
                                                             cov=self.f * self.H2[i] * self.cov)

        return res


class MLE:
    def __init__(self, d, alpha, size, show=True):
        self.show = show
        self.result = []
        if not self.show:
            self.Cache = []

        self.alpha = alpha
        aVar = np.array([alpha * (1 - alpha), 4 * (alpha * (1 - alpha)) ** 2])
        self.disp('Reference for a-var (prob) [direct, optimal]: {}'.format(np.round(aVar, 6)))

        self.T = lambda x: garch.target(x[:, :3], x[:, 3:])
        self.iP = lambda x: garch.proposal(x[:, :3], x[:, 3:])
        self.iS = lambda size: np.hstack(garch.predict(d, size))
        self.size = size

    def disp(self, text):
        if self.show:
            print(text)
        else:
            self.Cache.append(text)

    def __cumu(self, x):
        return x[:, 3:].sum(axis=1)

    def __divi(self, p, q):
        q[q == 0] = 1
        return p / q

    def __estimate(self, S, W, name, asym=True):
        x = self.__cumu(S)
        self.eVaR = quantile(x, W, self.alpha)
        self.result.append(self.eVaR)
        if asym:
            w = W / np.sum(W)
            aVar = np.sum((w * (1.0 * (x <= self.eVaR) - self.alpha)) ** 2) * x.size
            ESS = 1 / np.sum(w ** 2)
            Wf = W * (x <= self.eVaR)
            wf = Wf / np.sum(Wf)
            ESSf = 1 / np.sum(wf ** 2)
            self.disp('{} est: {:.4f}; a-var (prob): {:.6f}; ESS: {:.0f}/{}; ESS(f): {:.0f}/{}' \
                      .format(name, self.eVaR, aVar, ESS, x.size, ESSf, x.size))
        else:
            self.disp('{} est: {:.4f}'.format(name, self.eVaR))

        if any(W < 0):
            W[W < 0] = 0
            self.eVaR = quantile(x, W, self.alpha)
            self.disp('(adjusted) {} est: {:.4f}'.format(name, self.eVaR))

    def estimate_IS(self):
        S = self.iS(self.size)
        W = self.__divi(self.T(S), self.iP(S))
        self.__estimate(S, W, 'IS')

    def resampling(self, size, ratio):
        S = self.iS(ratio * size)
        W = self.__divi(self.T(S), self.iP(S))
        self.__estimate(S, W, 'IS({})'.format(ratio * size))
        p = W * np.abs(1.0 * (self.__cumu(S) <= self.eVaR) - self.alpha)
        index = np.arange(S.shape[0])
        self.choice = np.random.choice(index, size, p=p / np.sum(p), replace=True)

        self.rS = S[self.choice]
        self.rSset = S[list(set(self.choice))]
        self.disp('resampling rate: {}/{}'.format(self.rSset.shape[0], size))

    def clustering(self, seed=0, auto=False, num=4, draw=False, write=False):
        if auto:
            scaler = StandardScaler().fit(self.rS)
            kmeans = KMeans(n_clusters=num, random_state=seed).fit(scaler.transform(self.rS))
            self.rSs = [self.rS[kmeans.labels_ == i] for i in range(num)]
            nums = [len(set(self.choice[kmeans.labels_ == i])) for i in range(num)]
            self.disp('Clustering: {}/{}'.format(nums, [rS.shape[0] for rS in self.rSs]))
            self.group = lambda s: kmeans.predict(scaler.transform([s]))[0]
        else:
            rS1 = self.rS[self.__cumu(self.rS) <= self.eVaR]
            rS2 = self.rS[self.__cumu(self.rS) > self.eVaR]
            scaler1 = StandardScaler().fit(rS1)
            scaler2 = StandardScaler().fit(rS2)
            kmeans1 = KMeans(n_clusters=2, random_state=seed).fit(scaler1.transform(rS1))
            kmeans2 = KMeans(n_clusters=2, random_state=seed).fit(scaler2.transform(rS2))
            lb1 = kmeans1.labels_
            lb2 = kmeans2.labels_
            self.rSs = [rS1[lb1 == 1], rS1[lb1 == 0], rS2[lb2 == 1], rS2[lb2 == 0]]
            num1 = len(set(self.choice[self.__cumu(self.rS) <= self.eVaR][lb1 == 1]))
            num2 = len(set(self.choice[self.__cumu(self.rS) <= self.eVaR][lb1 == 0]))
            num3 = len(set(self.choice[self.__cumu(self.rS) > self.eVaR][lb2 == 1]))
            num4 = len(set(self.choice[self.__cumu(self.rS) > self.eVaR][lb2 == 0]))
            self.disp('Clustering: {}/{}, {}/{}, {}/{}, {}/{}' \
                      .format(num1, lb1.sum(), num2, (1 - lb1).sum(), num3, lb2.sum(), num4, (1 - lb2).sum()))
            tmp = np.copy(self.eVaR)
            def group(s):
                if s[3:].sum() <= tmp:
                    if kmeans1.predict(scaler1.transform([s]))[0] == 1:
                        return 0
                    else:
                        return 1
                else:
                    if kmeans2.predict(scaler2.transform([s]))[0] == 1:
                        return 2
                    else:
                        return 3

            self.group = group

        if draw:
            data = pd.DataFrame(self.rS, columns=['phi0', 'phi1', 'beta'] \
                                                 + ['y{}'.format(i + 1) for i in range(self.rS.shape[1] - 3)])
            data['type'] = [self.group(s) for s in self.rS]
            if write:
                data.to_csv('garch.csv', index=False)

            sb.pairplot(data, hue='type')
            plt.show()

    def estimate_NIS(self, rate, bw=1, a=1):
        kdes = []
        covs = []
        tmp = np.copy(self.eVaR)
        # F = lambda x: self.T(x) * np.abs(1.0 * (self.__cumu(x) <= tmp) - self.alpha)
        F = lambda x: np.ones(x.shape[0])
        for i, rS in enumerate(self.rSs):
            kdes.append(AKDE(rS,bw=bw,F=F(rS),a=a))
            covs.append(kdes[-1].f * kdes[-1].cov)
            self.disp('KDE {}: {} ({:.4f})' \
                      .format(i + 1, np.round(np.sqrt(np.diag(covs[-1])), 2), kdes[-1].f))

        rate0 = [rS.shape[0] / self.rS.shape[0] for rS in self.rSs]
        self.nP = lambda x: np.sum([r0 * kde.pdf(x) for r0, kde in zip(rate0, kdes)], axis=0)
        def nS(size):
            sizes = np.round(size * np.array(rate0)).astype(np.int64)
            sizes[-1] = size - sizes[:-1].sum()
            return np.vstack([kde.rvs(sz) for kde, sz in zip(kdes, sizes)])

        self.nS = nS
        def h(x,loc):
            f = F(np.array([loc]))[0]
            kde = kdes[self.group(loc)]
            cov = (f/kde.gm) ** (-2 * a) * covs[self.group(loc)]
            return mvnorm.pdf(x=x, mean=loc, cov=cov)

        self.G = lambda x: np.array([h(x, loc) for loc in self.rSset[1:]]) - self.nP(x)

        S = self.nS(self.size)
        W = self.__divi(self.T(S), self.nP(S))
        self.__estimate(S, W, 'NIS')

        self.mP = lambda x: (1 - rate) * self.iP(x) + rate * self.nP(x)
        self.mS = lambda size: np.vstack([self.iS(size - round(rate * size)), self.nS(round(rate * size))])
        self.S_ = self.mS(self.size)
        self.T_ = self.T(self.S_)
        self.mP_ = self.mP(self.S_)
        W = self.__divi(self.T_, self.mP_)
        self.__estimate(self.S_, W, 'MIS')
        self.G_ = self.G(self.S_)

    def estimate_RIS(self):
        X = (self.__divi(self.G_, self.mP_)).T
        tmp = X / np.linalg.norm(X, axis=0)
        lbd = np.linalg.eigvals(tmp.T.dot(tmp))
        tau = np.sqrt(lbd.max() / lbd)
        self.disp('Condition index: (min {:.4f}, median {:.4f}, mean {:.4f}, max {:.4f}, [>30] {}/{})' \
                  .format(tau.min(), np.median(tau), tau.mean(), tau.max(), np.sum(tau > 30), tau.size))

        y2 = self.__divi(self.T_, self.mP_)
        y1 = y2 * (self.__cumu(self.S_) <= self.eVaR)
        y3 = y1 - self.alpha * y2
        self.reg1 = Linear().fit(X, y1)
        self.reg2 = Linear().fit(X, y2)
        self.reg3 = Linear().fit(X, y3)
        self.disp('Tail R2: {:.4f}; Body R2: {:.4f}; Overall R2: {:.4f}' \
                  .format(self.reg1.score(X, y1), self.reg2.score(X, y2), self.reg3.score(X, y3)))

        W2 = y2 - X.dot(self.reg2.coef_)
        W3 = y3 - X.dot(self.reg3.coef_)
        aVar = W2.size * np.sum(W3 ** 2) / (np.sum(W2)) ** 2
        self.disp('RIS a-var: {:.6f}'.format(aVar))

        XX = X - X.mean(axis=0)
        self.zeta1 = np.linalg.solve(XX.T.dot(XX), X.sum(axis=0))
        W = self.__divi(self.T_, self.mP_) * (1 - XX.dot(self.zeta1))
        self.disp('reg weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                  .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
        self.__estimate(self.S_, W, 'RIS', asym=False)

    def estimate_MLE(self, opt=True, NR=True):
        target = lambda zeta: -np.mean(np.log(self.mP_ + zeta.dot(self.G_)))
        gradient = lambda zeta: -np.mean(self.__divi(self.G_, self.mP_ + zeta.dot(self.G_)), axis=1)
        hessian = lambda zeta: self.__divi(self.G_, (self.mP_ + zeta.dot(self.G_)) ** 2)\
                                   .dot(self.G_.T) / self.G_.shape[1]
        zeta0 = np.zeros(self.G_.shape[0])
        grad0 = gradient(zeta0)
        self.disp('MLE reference:')
        self.disp('origin: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                  .format(target(zeta0), grad0.min(), grad0.mean(), grad0.max(), grad0.std()))

        print()
        self.disp('Theoretical results:')
        self.disp('MLE(The) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})' \
                  .format(self.zeta1.min(), self.zeta1.mean(), self.zeta1.max(), \
                          self.zeta1.std(), np.sqrt(np.sum(self.zeta1 ** 2))))
        grad1 = gradient(self.zeta1)
        self.disp('theory: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                  .format(target(self.zeta1), grad1.min(), grad1.mean(), grad1.max(), grad1.std()))
        W = self.__divi(self.T_, self.mP_ + self.zeta1.dot(self.G_))
        self.disp('mle weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                  .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
        self.__estimate(self.S_, W, 'MLE(The)', asym=False)

        if opt:
            zeta = zeta0 if np.isnan(target(self.zeta1)) else self.zeta1
            begin = dt.now()
            if NR:
                res = root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta, method='lm', jac=True)
            else:
                cons = ({'type': 'ineq', 'fun': lambda zeta: self.mP_ + zeta.dot(self.G_), \
                         'jac': lambda zeta: self.G_.T})
                res = minimize(target, zeta, method='SLSQP', jac=gradient, constraints=cons, \
                               options={'ftol': 1e-8, 'maxiter': 1000})

            end = dt.now()
            print()
            self.disp('Optimization results (spent {} seconds):'.format((end - begin).seconds))
            if res['success']:
                zeta = res['x']
                self.disp('MLE(Opt) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})' \
                          .format(zeta.min(), zeta.mean(), zeta.max(), zeta.std(), np.sqrt(np.sum(zeta ** 2))))
                self.disp('Dist(zeta(Opt),zeta(The))={:.4f}'.format(np.sqrt(np.sum((zeta - self.zeta1) ** 2))))
                grad = gradient(zeta)
                self.disp('optimal: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                          .format(target(zeta), grad.min(), grad.mean(), grad.max(), grad.std()))
                W = self.__divi(self.T_, self.mP_ + zeta.dot(self.G_))
                self.disp('mle weights (Opt): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                          .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
                self.__estimate(self.S_, W, 'MLE(Opt)', asym=False)
            else:
                self.disp('MLE fail')


D = np.array([1, 2, 5])
Alpha = np.array([0.05, 0.01])
Truth = np.array([[-1.333, -1.895], [-1.886, -2.771], [-2.996, -4.424]])


def experiment(pars):
    np.random.seed(19971107)
    print('Start {} {}'.format(pars[0], pars[1]))
    mle = MLE(d=pars[0], alpha=pars[1], size=100000, show=True)
    mle.disp('Reference for VaR{} (d={}): {}'.format(pars[1], pars[0], Truth[D == pars[0], Alpha == pars[1]]))
    mle.disp('==IS==================================================IS==')
    mle.estimate_IS()
    mle.resampling(size=2000, ratio=1000)
    mle.disp('==NIS================================================NIS==')
    mle.clustering(auto=False, num=4, draw=True, write=False)
    mle.estimate_NIS(rate=0.9)
    mle.disp('==RIS================================================RIS==')
    mle.estimate_RIS()
    # mle.disp('==MLE================================================MLE==')
    # mle.estimate_MLE()
    print('End {} {}'.format(pars[0], pars[1]))
    # return mle.Cache


def main():
    begin = dt.now()
    # Cache = []
    # for d in D:
    #     for alpha in Alpha:
    #         Cache.append(experiment((d, alpha)))
    experiment((2,0.05))
    end = dt.now()
    print((end - begin).seconds)
    # return Cache


if __name__ == '__main__':
    main()
    # result = main()
    # with open('Ian', 'wb') as file:
    #     pickle.dump(result, file)
    #     file.close()

