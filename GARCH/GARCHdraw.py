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

from scipy.stats import norm,t,truncnorm
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import multivariate_t as mvt
from scipy.spatial import Delaunay as TRI
from scipy.interpolate import LinearNDInterpolator as ITP
from scipy.optimize import minimize,root
from scipy.optimize import NonlinearConstraint as NonlinCons
from scipy.stats import gaussian_kde as sciKDE

from sklearn.linear_model import LinearRegression as Linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KernelDensity as sklKDE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

data=DR('^GSPC','yahoo',dt(2010,9,29),dt(2011,7,14))
returns=pd.DataFrame(100*np.diff(np.log(data['Adj Close'])),columns=['dlr'])
returns.index=data.index.values[1:data.index.values.shape[0]]
returns=np.array(returns['dlr'])

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

garch=GARCH(returns)
garch.laplace(inflate=2,df=1)

class MLE:
    def __init__(self, d, alpha, size, show=True):
        self.show = show
        if not self.show:
            self.Cache = []

        self.alpha = alpha
        aVar = np.array([alpha * (1 - alpha), 4 * (alpha * (1 - alpha)) ** 2])
        self.disp('Reference for a-var (prob) [direct, optimal]: {}'.format(np.round(aVar, 6)))

        self.T = lambda x: garch.target(x[:, :3], x[:, 3:])
        self.iP = lambda x: garch.proposal(x[:, :3], x[:, 3:])
        self.iS = lambda size: np.hstack(garch.predict(d, size))
        self.oP = lambda x, VaR: self.T(x) * np.abs(1.0 * (self.__cumu(x) < VaR) - self.alpha) / (
                    2 * self.alpha * (1 - self.alpha))
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

    def resample(self, size, ratio):
        S = self.iS(ratio * size)
        p = self.__divi(self.oP(S, self.eVaR), self.iP(S))
        index = np.arange(S.shape[0])
        self.choice = np.random.choice(index, size, p=p / np.sum(p), replace=True)

        self.rS = S[self.choice]
        self.rSset = S[list(set(self.choice))]
        self.disp('resampling rate: {}/{}'.format(self.rSset.shape[0], size))

    def cluster(self, seed=0):
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

    def estimate_NIS(self, rate, bdwth='silverman'):
        kdes = []
        covs = []
        for i, rS in enumerate(self.rSs):
            kdes.append(sciKDE(rS.T, bw_method=bdwth))
            covs.append(kdes[-1].covariance_factor() * np.cov(rS.T))
            self.disp('KDE {}: {} ({:.4f})' \
                      .format(i + 1, np.round(np.sqrt(np.diag(covs[-1])), 2), kdes[-1].covariance_factor()))

        self.h = lambda x, loc: mvnorm.pdf(x=x, mean=loc, cov=covs[self.group(loc)])
        self.G = lambda x: np.array([self.h(x, loc) for loc in self.rSset[1:]]) - self.nP(x)
        rate0 = [rS.shape[0] / self.rS.shape[0] for rS in self.rSs]
        self.nP = lambda x: np.sum([r0 * kde.pdf(x.T) for r0, kde in zip(rate0, kdes)], axis=0)

        def nS(size):
            sizes = np.round(size * np.array(rate0)).astype(np.int)
            sizes[-1] = size - sizes[:-1].sum()
            return np.vstack([kde.resample(sz).T for kde, sz in zip(kdes, sizes)])

        self.nS = nS

        S = self.nS(self.size)
        W = self.__divi(self.T(S), self.nP(S))
        self.__estimate(S, W, 'NIS')

        self.mP = lambda x: (1 - rate) * self.iP(x) + rate * self.nP(x)
        self.mS = lambda size: np.vstack([self.iS(size - round(rate * size)), self.nS(round(rate * size))])
        self.S = self.mS(self.size)
        W = self.__divi(self.T(self.S), self.mP(self.S))
        self.__estimate(self.S, W, 'MIS')

    def estimate_RIS(self):
        T = self.T(self.S)
        mP = self.mP(self.S)
        X = (self.__divi(self.G(self.S), mP)).T
        tmp = X / np.linalg.norm(X, axis=0)
        lbd = np.linalg.eigvals(tmp.T.dot(tmp))
        tau = np.sqrt(lbd.max() / lbd)
        self.disp('Condition index: (min {:.4f}, median {:.4f}, mean {:.4f}, max {:.4f}, [>30] {}/{})' \
                  .format(tau.min(), np.median(tau), tau.mean(), tau.max(), np.sum(tau > 30), tau.size))

        y2 = self.__divi(T, mP)
        y1 = y2 * (self.__cumu(self.S) <= self.eVaR)
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
        zeta1 = np.linalg.solve(XX.T.dot(XX), X.sum(axis=0))
        W = self.__divi(self.T(self.S), mP) * (1 - XX.dot(zeta1))
        self.disp('reg weights: (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                  .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
        self.__estimate(self.S, W, 'RIS', asym=False)

    def estimate_MLE(self, opt=True, NR=True):
        mP = self.mP(self.S)
        G = self.G(self.S)
        target = lambda zeta: -np.mean(np.log(mP + zeta.dot(G)))
        gradient = lambda zeta: -np.mean(self.__divi(G, mP + zeta.dot(G)), axis=1)
        hessian = lambda zeta: self.__divi(G, (mP + zeta.dot(G)) ** 2).dot(G.T) / G.shape[1]
        zeta0 = np.zeros(G.shape[0])
        grad0 = gradient(zeta0)
        self.disp('MLE reference:')
        self.disp('origin: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                  .format(target(zeta0), grad0.min(), grad0.mean(), grad0.max(), grad0.std()))

        print()
        self.disp('Theoretical results:')
        X = self.__divi(G, mP).T
        XX = X - X.mean(axis=0)
        zeta1 = np.linalg.solve(XX.T.dot(XX), X.sum(axis=0))
        self.disp('MLE(The) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})' \
                  .format(zeta1.min(), zeta1.mean(), zeta1.max(), zeta1.std(), np.sqrt(np.sum(zeta1 ** 2))))
        grad1 = gradient(zeta1)
        self.disp('theory: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                  .format(target(zeta1), grad1.min(), grad1.mean(), grad1.max(), grad1.std()))
        W = self.__divi(self.T(self.S), mP + zeta1.dot(G))
        self.disp('mle weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                  .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
        self.__estimate(self.S, W, 'MLE(The)', asym=False)

        if opt:
            zeta = zeta1 if target(zeta1) != np.nan else zeta0
            begin = dt.now()
            if NR:
                res = root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta, method='lm', jac=True)
            else:
                cons = ({'type': 'ineq', 'fun': lambda zeta: mP + zeta.dot(G), 'jac': lambda zeta: G.T})
                res = minimize(target, zeta, method='SLSQP', jac=gradient, constraints=cons, \
                               options={'ftol': 1e-8, 'maxiter': 1000})

            end = dt.now()
            print()
            self.disp('Optimization results (spent {} seconds):'.format((end - begin).seconds))
            if res['success']:
                zeta = res['x']
                self.disp('MLE(Opt) zeta: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f}, norm {:.4f})' \
                          .format(zeta.min(), zeta.mean(), zeta.max(), zeta.std(), np.sqrt(np.sum(zeta ** 2))))
                self.disp('Dist(zeta(Opt),zeta(The))={:.4f}'.format(np.sqrt(np.sum((zeta - zeta1) ** 2))))
                grad = gradient(zeta)
                self.disp('optimal: value: {:.4f}; grad: (min {:.4f}, mean {:.4f}, max {:.4f}, std {:.4f})' \
                          .format(target(zeta), grad.min(), grad.mean(), grad.max(), grad.std()))
                W = self.__divi(self.T(self.S), mP + zeta.dot(G))
                self.disp('mle weights (Opt): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})' \
                          .format(W.min(), W.mean(), W.max(), np.sum(W < 0), W.size))
                self.__estimate(self.S, W, 'MLE(Opt)', asym=False)
            else:
                self.disp('MLE fail')

D=np.array([1,2,5])
Alpha=np.array([0.05,0.01])
Truth=np.array([[-1.333,-1.895],[-1.886,-2.771],[-2.996,-4.424]])

def main():
    mle = MLE(d=2, alpha=0.01, size=100000, show=True)
    mle.estimate_IS()
    mle.resample(size=2000, ratio=1000)
    mle.cluster()

    data = pd.DataFrame(mle.rS, columns=['phi0', 'phi1', 'beta', 'y1', 'y2'])
    data['type'] = [mle.group(s) for s in mle.rS]
    data.to_csv('garch.csv', index=False)
    sb.pairplot(data, hue='type',palette={0:'red',1:'blue',2:'green',3:'yellow'})
    plt.show()

if __name__ == '__main__':
    main()
