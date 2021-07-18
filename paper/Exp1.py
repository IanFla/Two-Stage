import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt

from scipy.stats import uniform
from scipy.stats import multivariate_normal as mvnorm
from scipy.optimize import minimize, root
from scipy.stats import gmean

from sklearn.linear_model import LinearRegression as Linear
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

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
        return np.argsort(distances)[:np.around(self.neff).astype(np.int64)]

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

        samples = np.zeros([size, self.d])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.lambda2s[j] * self.covs
            samples[cum_sizes[j]:cum_sizes[j + 1]] = mvnorm.rvs(size=sizes[j], mean=center, cov=cov)

        return samples


class MLE:
    def __init__(self, dim, target, init_proposal, size_est, show=True):
        self.show = show
        self.cache = []
        self.result = []
        self.dim = dim

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

        if any(weights < 0) and check:
            weights[weights < 0] = 0
            Z = np.mean(weights)
            Err = np.abs(Z - 1)
            print('{} est (Adjusted): {:.4f}; err: {:.4f}'.format(name, Z, Err))

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

    def proposal(self, bw=1.0, local=False, gamma=0.1, a=0.0, rate=0.9):
        self.kde = KDE(self.centers, self.weights, bw=bw, local=local, gamma=gamma, ps=self.ps, a=a)
        covs = self.kde.covs.mean(axis=0) if local else self.kde.covs
        bdwth = np.mean(np.sqrt(np.diag(covs)))
        self.disp('KDE: (factor {:.4f}, bdwth: {:.4f}, ESS {:.0f}/{})'
                  .format(self.kde.factor, bdwth, self.kde.neff, self.weights.size))
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

            return np.array(out) - self.mix_proposal(x)

        self.controls = controls

    def nonparametric_estimation(self):
        samples = self.nonpar_sampler(self.size_est)
        weights = self.__divi(self.target(samples), self.nonpar_proposal(samples))
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

        y = self.weights_
        self.regO = Linear().fit(X, y)
        self.regR = Ridge(alpha=alphaR).fit(X, y)
        self.regL = Lasso(alpha=alphaL).fit(X, y)
        print('Ordinary R2: {:.4f}; Ridge R2: {:.4f}; Lasso R2: {:.4f}'
              .format(self.regO.score(X, y), self.regR.score(X, y), self.regL.score(X, y)))

        weights = y - X.dot(self.regO.coef_)
        self.__estimate(weights, 'RIS(Ord)', check=False)
        weights = y - X.dot(self.regR.coef_)
        self.__estimate(weights, 'RIS(Rid)', check=False)
        weights = y - X.dot(self.regL.coef_)
        self.__estimate(weights, 'RIS(Las)', check=False)

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

        print()
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
            print()
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

        ax.set_title('{}-D target and {} proposal ({}thD slicing)'.format(self.dim, name, dim + 1))
        plt.show()


def experiment(seed, dim, size_est,
               size, ratio, resample,
               bw, local, gamma, a, rate,
               alphaR, alphaL):
    np.random.seed(seed)
    mean = np.zeros(dim)
    target = mvnorm(mean=mean)
    init_proposal = mvnorm(mean=mean, cov=4)
    mle = MLE(dim, target, init_proposal, size_est=size_est, show=True)
    mle.disp('==IS==================================================IS==')
    mle.initial_estimation()
    x = np.linspace(-4, 4, 101)
    mle.draw(mle.init_proposal, x=x, name='initial')
    mle.resampling(size=size, ratio=ratio, resample=resample)
    mle.disp('==NIS================================================NIS==')
    mle.proposal(bw=bw, local=local, gamma=gamma, a=a, rate=rate)
    mle.nonparametric_estimation()
    mle.draw(mle.nonpar_proposal, x=x, name='nonparametric')
    mle.disp('==RIS================================================RIS==')
    mle.regression_estimation(alphaR=alphaR, alphaL=alphaL)
    mle.draw(mle.mix_proposal, x=x, name='regression')
    mle.disp('==MLE================================================MLE==')
    mle.likelihood_estimation(opt=True, NR=True)


def main():
    begin = dt.now()
    # experiment(seed=1234, dim=8, size_est=100000,
    #            size=1000, ratio=100, resample=True,
    #            bw=1.4, local=False, gamma=0.1, a=0.0, rate=0.9,
    #            alphaR=1000000.0, alphaL=0.1)

    # experiment(seed=1234, dim=8, size_est=100000,
    #            size=1000, ratio=100, resample=False,
    #            bw=1.4, local=False, gamma=0.1, a=0.0, rate=0.9,
    #            alphaR=1000000.0, alphaL=0.1)

    # experiment(seed=1234, dim=8, size_est=100000,
    #            size=1000, ratio=100, resample=True,
    #            bw=1.8, local=True, gamma=0.3, a=0.0, rate=0.9,
    #            alphaR=1000000.0, alphaL=0.1)

    # experiment(seed=1234, dim=8, size_est=100000,
    #            size=1000, ratio=100, resample=True,
    #            bw=2.0, local=False, gamma=0.1, a=1/8, rate=0.9,
    #            alphaR=1000000.0, alphaL=0.1)

    # experiment(seed=1234, dim=8, size_est=100000,
    #            size=1000, ratio=100, resample=True,
    #            bw=5.0, local=False, gamma=0.1, a=1/2, rate=0.9,
    #            alphaR=1000000.0, alphaL=0.1)

    end = dt.now()
    print('Total spent: {}s'.format((end - begin).seconds))


if __name__ == '__main__':
    main()
