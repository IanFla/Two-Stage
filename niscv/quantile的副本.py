import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
from particles import resampling as rs
# import pickle
# import multiprocessing

from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import multivariate_t as mvt
from scipy.optimize import minimize, root
from scipy.stats import gmean
from scipy.stats import gamma as Ga
from scipy.spatial.distance import mahalanobis

from sklearn.linear_model import LinearRegression as Linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import warnings
warnings.filterwarnings("ignore")


class MLE:
    def __init__(self, dim, target, indicator, alpha, init_proposal, size_est, show=True):
        self.show = show
        self.cache = []
        self.result = []
        self.dim = dim

        self.target = target.pdf
        self.indicator = indicator
        self.alpha = alpha
        self.init_proposal = init_proposal.pdf
        self.init_sampler = init_proposal.rvs
        self.size_est = size_est

        self.centers = None
        self.weights = None

        self.kde = None
        self.nonpar_proposal = None
        self.nonpar_sampler = None
        self.mix_proposal = None
        self.mix_sampler = None
        self.controls = None

        self.samples_ = None
        self.target_ = None
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

    def __estimate(self, weights, name, asym=True, check=True):
        Prob = weights.mean()
        self.result.append(Prob)
        Err = np.abs(Prob - self.alpha)
        if asym:
            aVar = np.mean((weights - self.alpha) ** 2)
            self.result.append(aVar)
            aErr = np.sqrt(aVar / weights.size)
            ESS = 1 / np.sum((weights / np.sum(weights)) ** 2)
            self.disp('{} est: {:.4f}; err: {:.4f}; a-var: {:.4f}; a-err: {:.4f}; ESS: {:.0f}/{}'
                      .format(name, Prob, Err, aVar, aErr, ESS, weights.size))
        else:
            self.disp('{} est: {:.4f}; err: {:.4f}'.format(name, Prob, Err))

        if any(weights < 0) and check:
            weights[weights < 0] = 0
            Prob = weights.mean()
            Err = np.abs(Prob - self.alpha)
            self.disp('{} est (Adjusted): {:.4f}; err: {:.4f}'.format(name, Prob, Err))

    def initial_estimation(self):
        samples = self.init_sampler(self.size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples)) * self.indicator(samples)
        self.__estimate(weights, 'IS')

        ESS = 1 / np.sum((weights / weights.sum()) ** 2)
        RSS = weights.sum() / weights.max()
        self.disp('Ratio reference: n0/ESS {:.0f} ~ n0/RSS {:.0f}'.format(self.size_est / ESS, self.size_est / RSS))
        self.result.extend([self.size_est / ESS, self.size_est / RSS])

    def resampling(self, size, ratio, resample=True):
        if resample:
            samples = self.init_sampler(ratio * size)
            weights = self.__divi(self.target(samples), self.init_proposal(samples)) * self.indicator(samples)
            index, sizes = np.unique(rs.systematic(weights / weights.sum(), M=size), return_counts=True)
            self.centers = samples[index]
            self.weights = sizes
            self.disp('Resampling rate: {}/{}'.format(self.weights.size, size))
            self.result.append(self.weights.size)
        else:
            self.centers = self.init_sampler(size)
            self.weights = self.__divi(self.target(self.centers),
                                       self.init_proposal(self.centers)) * self.indicator(self.centers)

    def proposal(self, bw=1.0, factor='scott', local=False, gamma=0.3, kdf=0, alpha0=0.1):
        self.kde = KDE(self.centers, self.weights, bw=bw, factor=factor, local=local, gamma=gamma, kdf=kdf)
        covs = self.kde.covs.mean(axis=0) if local else self.kde.lambda2s.mean() * self.kde.covs
        bdwths = np.sqrt(np.diag(covs))
        self.disp('KDE: (mean bdwth: {:.4f}, factor {:.4f}, ESS {:.0f}/{})'
                  .format(np.mean(bdwths), self.kde.factor, self.kde.neff, self.kde.size))
        self.result.extend([np.mean(bdwths), self.kde.neff])
        self.nonpar_proposal = self.kde.pdf
        self.nonpar_sampler = self.kde.rvs
        self.mix_proposal = lambda x: alpha0 * self.init_proposal(x) + (1 - alpha0) * self.nonpar_proposal(x)
        self.mix_sampler = lambda size: np.vstack([self.init_sampler(round(alpha0 * size)),
                                                   self.nonpar_sampler(size - round(alpha0 * size))])

        def controls(x):
            out = np.zeros([self.centers.shape[0], x.shape[0]])
            for j, center in enumerate(self.centers):
                cov = self.kde.covs[j] if local else self.kde.lambda2s[j] * self.kde.covs
                out[j] = self.kde.kernel_pdf(x=x, m=center, v=cov)

            return out - self.mix_proposal(x)

        self.controls = controls

    def nonparametric_estimation(self, Rf=1.0):
        samples = self.nonpar_sampler(self.size_est)
        target = self.target(samples)
        proposal = self.nonpar_proposal(samples)
        ISE = np.mean(proposal - 2 * target * self.indicator(samples)) + Rf
        weights = self.__divi(target, proposal) * self.indicator(samples)
        KLD = np.mean(weights * np.log(weights + 1.0 * (weights == 0)))
        self.disp('sqrt(ISE/Rf): {:.4f}; KLD: {:.4f}'.format(np.sqrt(ISE/Rf), KLD))
        self.result.extend([np.sqrt(ISE / Rf), KLD])
        self.__estimate(weights, 'NIS')

        self.samples_ = self.mix_sampler(self.size_est)
        self.target_ = self.target(self.samples_)
        self.proposal_ = self.mix_proposal(self.samples_)
        self.weights_ = self.__divi(self.target_, self.proposal_) * self.indicator(self.samples_)
        self.__estimate(self.weights_, 'MIS')

    def regression_estimation(self, alphaR, alphaL):
        self.controls_ = self.controls(self.samples_)
        X = (self.__divi(self.controls_, self.proposal_)).T
        standard_X = X / np.linalg.norm(X, axis=0)
        lbds = np.linalg.eigvals(standard_X.T.dot(standard_X))
        del standard_X
        etas = np.sqrt(lbds.max(initial=0) / lbds)
        self.disp('Condition index: (min {:.4f}, median {:.4f}, mean {:.4f}, max {:.4f}, [>30] {}/{})'
                  .format(etas.min(), np.median(etas), etas.mean(), etas.max(), np.sum(etas > 30), etas.size))
        self.result.append(np.sum(etas > 30))

        y = self.weights_
        self.regO = Linear().fit(X, y)
        self.regR = Ridge(alpha=alphaR).fit(X, y)
        self.regL = Lasso(alpha=alphaL).fit(X, y)
        self.disp('Ordinary R2: {:.4f}; Ridge R2: {:.4f}; Lasso R2: {:.4f}'
                  .format(self.regO.score(X, y), self.regR.score(X, y), self.regL.score(X, y)))
        self.result.extend([self.regO.score(X, y), self.regR.score(X, y), self.regL.score(X, y)])

        weights = y - X.dot(self.regO.coef_)
        self.__estimate(weights, 'RIS(Ord)', check=False)
        weights = y - X.dot(self.regR.coef_)
        self.__estimate(weights, 'RIS(Rid)', check=False)
        weights = y - X.dot(self.regL.coef_)
        self.__estimate(weights, 'RIS(Las)', check=False)

        del X, y, weights
        samples = self.mix_sampler(self.size_est)
        proposal = self.mix_proposal(samples)
        y2 = self.__divi(self.target(samples), proposal) * self.indicator(samples)
        X2 = self.__divi(self.controls(samples), proposal).T
        weights = y2 - X2.dot(self.regO.coef_)
        self.__estimate(weights, 'RIS(Ord, unbiased)', check=False)
        weights = y2 - X2.dot(self.regR.coef_)
        self.__estimate(weights, 'RIS(Rid, unbiased)', check=False)
        weights = y2 - X2.dot(self.regL.coef_)
        self.__estimate(weights, 'RIS(Las, unbiased)', check=False)

    def likelihood_estimation(self, opt=True, NR=True):
        target = lambda zeta: -np.mean(np.log(self.proposal_ + zeta.dot(self.controls_)))
        gradient = lambda zeta: -np.mean(self.__divi(self.controls_, self.proposal_ + zeta.dot(self.controls_)), axis=1)
        hessian = lambda zeta: self.__divi(self.controls_, (self.proposal_ + zeta.dot(self.controls_))
                                           ** 2).dot(self.controls_.T) / self.controls_.shape[1]
        zeta0 = np.zeros(self.controls_.shape[0])
        grad0 = gradient(zeta0)
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
        self.__estimate(weights, 'RIS(The)', asym=False)
        weights = self.__divi(self.target_ * self.indicator(self.samples_), self.proposal_ + zeta1.dot(self.controls_))
        self.disp('MLE weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                  .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
        self.__estimate(weights, 'MLE(The)', asym=False)

        if opt:
            zeta2 = zeta0 if np.isnan(target(zeta1)) else zeta1
            begin = dt.now()
            if NR:
                res = root(lambda zeta: (gradient(zeta), hessian(zeta)), zeta2, method='lm', jac=True)
            else:
                cons = ({'type': 'ineq', 'fun': lambda zeta: self.proposal_ + zeta.dot(self.controls_),
                         'jac': lambda zeta: self.controls_.T})
                res = minimize(target, zeta2, method='SLSQP', jac=gradient, constraints=cons,
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
                weights = self.__divi(self.target_ * self.indicator(self.samples_),
                                      self.proposal_ + zeta2.dot(self.controls_))
                self.disp('MLE weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                          .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
                self.__estimate(weights, 'MLE(Opt)', asym=False)
            else:
                self.disp('MLE fail')

    def draw(self, proposal, x, name, dim=0):
        X = np.zeros([x.size, self.dim])
        X[:, dim] = x
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, self.target(X) * self.indicator(X))
        ax.plot(x, proposal(X))
        if name == 'nonparametric':
            ax.plot(x, self.mix_proposal(X))
            ax.legend(['optimal proposal', 'nonparametric proposal', 'mixture proposal'])
        elif name == 'regression':
            controls_ = self.controls(X)
            proposalO = self.regO.coef_.dot(controls_) + self.regO.intercept_ * proposal(X)
            proposalR = self.regR.coef_.dot(controls_) + self.regR.intercept_ * proposal(X)
            proposalL = self.regL.coef_.dot(controls_) + self.regL.intercept_ * proposal(X)
            ax.plot(x, proposalO)
            ax.plot(x, proposalR)
            ax.plot(x, proposalL)
            ax.legend(['optimal proposal', 'mixture proposal', 'ordinary regression',
                       'ridge regression', 'lasso regression'])
        else:
            ax.legend(['optimal proposal', '{} proposal'.format(name)])

        ax.set_title('{}-D optimal proposal and {} proposal ({}th slicing)'.format(self.dim, name, dim + 1))
        plt.show()


def experiment(seed, dim, target, indicator, alpha,
               init_proposal, size_est, x,
               size, ratio, resample,
               bw, factor, local, gamma, kdf, alpha0,
               alphaR, alphaL,
               stage=4, show=False):
    np.random.seed(seed)
    mle = MLE(dim, target, indicator, alpha, init_proposal, size_est=size_est, show=show)
    if stage >= 1:
        mle.disp('==IS==================================================IS==')
        mle.initial_estimation()
        if mle.show:
            mle.draw(mle.init_proposal, x=x, name='initial')

        mle.resampling(size=size, ratio=ratio, resample=resample)
        if stage >= 2:
            mle.disp('==NIS================================================NIS==')
            mle.proposal(bw=bw, factor=factor, local=local, gamma=gamma, kdf=kdf, alpha0=alpha0)
            Rf = target.pdf(target.rvs(size=size_est, random_state=seed)).mean()
            mle.nonparametric_estimation(Rf=Rf)
            if mle.show:
                mle.draw(mle.nonpar_proposal, x=x, name='nonparametric')

            if stage >= 3:
                mle.disp('==RIS================================================RIS==')
                mle.regression_estimation(alphaR=alphaR, alphaL=alphaL)
                if mle.show:
                    mle.draw(mle.mix_proposal, x=x, name='regression')

                if stage >= 4:
                    mle.disp('==MLE================================================MLE==')
                    mle.likelihood_estimation(opt=True, NR=True)

    return mle.result


def run(inputs):
    begin = dt.now()
    mean = np.zeros(5)
    target = mvnorm(mean=mean)
    L = 2 ** 2
    U = 2.5 ** 2

    def indicator(X):
        X2 = np.sum(X ** 2, axis=1)
        return (L <= X2) & (X2 <= U)

    cdf = Ga(a=mean.size/2, scale=2).cdf
    alpha = cdf(U) - cdf(L)
    init_proposal = mvnorm(mean=mean, cov=4)
    x = np.linspace(-4, 4, 101)
    result = experiment(seed=19971107, dim=mean.size, target=target, indicator=indicator, alpha=alpha,
                        init_proposal=init_proposal, size_est=50000, x=x,
                        size=500, ratio=100, resample=True,
                        bw=inputs[0], factor='scott', local=True, gamma=0.4, kdf=0, alpha0=0.1,
                        alphaR=10000.0, alphaL=0.1,
                        stage=3, show=True)
    end = dt.now()
    print('Total spent: {}s (bw {:.2f})'
          .format((end - begin).seconds, inputs[0]))
    return inputs + result


def main():
    run([1.0])


if __name__ == '__main__':
    main()
