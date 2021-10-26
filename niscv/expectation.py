import numpy as np
from matplotlib import pyplot as plt
from particles import resampling as rs
import scipy.stats as st
import scipy.optimize as opt
from kde import KDE
import sklearn.linear_model as lmd

from datetime import datetime as dt
import pickle
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


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
            avar = np.sum((weights * (funs - mu) / np.sum(weights)) ** 2) if self.sn else np.var(weights * funs)
            self.result.append(avar)
            aerr = np.sqrt(avar / weights.size)
            self.disp('{} est: {:.4f}; a-var: {:.4f}; a-err: {:.4f}'.format(name, mu, avar, aerr))
        else:
            self.disp('{} est: {:.4f}'.format(name, mu))

    def initial_estimation(self, size_kn, ratio, resample=True):
        size_est = ratio * size_kn
        samples = self.init_sampler(size_est)
        weights = self.__divi(self.target(samples), self.init_proposal(samples))
        funs = self.fun(samples)
        self.__estimate(weights, funs, 'IS')

        mu = self.result[-2]
        opt_proposal = (lambda x: self.target(x) * np.abs(self.fun(x) - mu)) \
            if self.sn else (lambda x: self.target(x) * np.abs(self.fun(x)))

        if resample:
            weights_kn = self.__divi(opt_proposal(samples), self.init_proposal(samples))
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
            self.weights_kn = self.__divi(opt_proposal(self.centers), self.init_proposal(self.centers))

    def proposal(self, bw=1.0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1):
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

    def nonparametric_estimation(self, Rf=1.0):
        samples = self.nonpar_sampler(self.size_est)
        target = self.target(samples)
        proposal = self.nonpar_proposal(samples)
        ISE = np.mean(proposal - 2 * target) + Rf
        weights = self.__divi(target, proposal)
        KLD = np.mean(weights * np.log(weights + 1.0 * (weights == 0)))
        self.disp('sqrt(ISE/Rf): {:.4f}; KLD: {:.4f}'.format(np.sqrt(ISE/Rf), KLD))
        self.result.extend([np.sqrt(ISE / Rf), KLD])
        self.__estimate(weights, 'NIS')

        self.samples_ = self.mix_sampler(self.size_est)
        self.target_ = self.target(self.samples_)
        self.proposal_ = self.mix_proposal(self.samples_)
        self.weights_ = self.__divi(self.target_, self.proposal_)
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
        y2 = self.__divi(self.target(samples), proposal)
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
        weights = self.__divi(self.target_, self.proposal_ + zeta1.dot(self.controls_))
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
                weights = self.__divi(self.target_, self.proposal_ + zeta2.dot(self.controls_))
                self.disp('MLE weights (The): (min {:.4f}, mean {:.4f}, max {:.4f}, [<0] {}/{})'
                          .format(weights.min(), weights.mean(), weights.max(), np.sum(weights < 0), weights.size))
                self.__estimate(weights, 'MLE(Opt)', asym=False)
            else:
                self.disp('MLE fail')

    def draw(self, proposal, x, name, dim=0):
        X = np.zeros([x.size, self.dim])
        X[:, dim] = x
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, self.target(X))
        ax.plot(x, proposal(X))
        if name == 'nonparametric':
            ax.plot(x, self.mix_proposal(X))
            ax.legend(['target', 'nonparametric proposal', 'mixture proposal'])
        elif name == 'regression':
            controls_ = self.controls(X)
            proposalO = self.regO.coef_.dot(controls_) + self.regO.intercept_ * proposal(X)
            proposalR = self.regR.coef_.dot(controls_) + self.regR.intercept_ * proposal(X)
            proposalL = self.regL.coef_.dot(controls_) + self.regL.intercept_ * proposal(X)
            ax.plot(x, proposalO)
            ax.plot(x, proposalR)
            ax.plot(x, proposalL)
            ax.legend(['target', 'mixture proposal', 'ordinary regression', 'ridge regression', 'lasso regression'])
        else:
            ax.legend(['target', '{} proposal'.format(name)])

        ax.set_title('{}-D target and {} proposal ({}th slicing)'.format(self.dim, name, dim + 1))
        plt.show()


def experiment(seed, dim, target,
               init_proposal, size_est, x,
               size, ratio, resample,
               bw, factor, local, gamma, kdf, alpha0,
               alphaR, alphaL,
               stage=4, show=False):
    np.random.seed(seed)
    mle = MLE(dim, target, init_proposal, size_est=size_est, show=show)
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
    mean = np.zeros(inputs[0])
    target = mvnorm(mean=mean)
    init_proposal = mvnorm(mean=mean, cov=4)
    x = np.linspace(-4, 4, 101)
    print(inputs)
    result = experiment(seed=3033079628, dim=mean.size, target=target,
                        init_proposal=init_proposal, size_est=100000, x=x,
                        size=inputs[1], ratio=100, resample=True,
                        bw=1.0, factor='scott', local=False, gamma=1.0, kdf=0, alpha0=0.1,
                        alphaR=10000.0, alphaL=0.1,
                        stage=4, show=False)
    end = dt.now()
    print('Total spent: {}s (dim {} size {})'
          .format((end - begin).seconds, inputs[0], inputs[1]))
    return inputs + result


def main():
    Dim = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    Size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 450, 500]
    inputs = []
    for dim in Dim:
        for size in Size:
            inputs.append([dim, size])

    pool = multiprocessing.Pool(2)
    results = pool.map(run, inputs)

    with open('DimSize', 'wb') as file:
        pickle.dump(results, file)
        file.close()


if __name__ == '__main__':
    main()
