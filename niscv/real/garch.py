import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as opt
import numdifftools as nd
import warnings

warnings.filterwarnings("ignore")


class GARCH:
    def __init__(self):
        df = pd.read_csv('../data/garch/SP500.csv')
        data = df.VALUE.values[1:] - df.VALUE.values[:-1]
        ys = 100 * data[2700:2900]
        self.h0 = np.std(ys)
        self.y0 = ys[0]
        self.y1toT = ys[1:]
        self.T = self.y1toT.size
        self.prior_pars = [-1, 2]
        self.rvs_trunc = None
        self.pdf_trunc = None

    def posterior(self, pars):
        neglogpdfp0toT = 0.5 * ((pars[:, 0] - self.prior_pars[0]) / self.prior_pars[1]) ** 2
        h = np.exp(pars[:, 0]) + pars[:, 1] * self.y0 ** 2 + pars[:, 2] * self.h0
        for i in range(self.T):
            neglogpdfp0toT += 0.5 * (self.y1toT[i] ** 2 / h + np.log(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * self.y1toT[i] ** 2 + pars[:, 2] * h

        return neglogpdfp0toT - 57.0

    @staticmethod
    def __test(pars):
        return (pars[:, 1] >= 0) & (pars[:, 2] >= 0) & (pars[:, 1] + pars[:, 2] < 1)

    def laplace(self, inflate=2, df=1, p_acc=0.48194609111):
        cons = ({'type': 'ineq',
                 'fun': lambda pars: np.array([pars[1], pars[2], 1 - pars[1] - pars[2]]),
                 'jac': lambda x: np.array([[0, 1, 0], [0, 0, 1], [0, -1, -1]])})
        target = lambda pars: self.posterior(pars.reshape([1, -1]))
        mu0 = np.array([0, 0.1, 0.7])
        res = opt.minimize(target, mu0, method='SLSQP', constraints=cons,
                           options={'maxiter': 1000, 'ftol': 1e-100, 'gtol': 1e-100, 'disp': False})
        mu = res['x']
        Sigma = np.linalg.inv(nd.Hessian(target)(mu))
        Sigma[:, 0] *= inflate
        Sigma[0, :] *= inflate
        pdf_full = lambda pars: np.prod([st.t.pdf(x=pars[:, i], df=df, loc=mu[i],
                                                  scale=np.sqrt(Sigma[i, i])) for i in range(3)], axis=0)
        rvs_full = lambda size: np.array([st.t.rvs(size=size, df=df, loc=mu[i],
                                                   scale=np.sqrt(Sigma[i, i])) for i in range(3)]).T

        def estimate(size):
            samples = rvs_full(size)
            p = np.mean(self.__test(samples))
            err = np.sqrt(p * (1 - p) / size)
            return p, [p - 2 * err, p + 2 * err]

        def pdf_trunc(pars):
            pdf = pdf_full(pars)
            good = self.__test(pars)
            return good * pdf / p_acc

        def rvs_trunc(size):
            pars = rvs_full(int(2 * size / p_acc))
            good = self.__test(pars)
            return pars[good][:size]

        self.pdf_trunc = pdf_trunc
        self.rvs_trunc = rvs_trunc
        return estimate

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
            ypre[:, i] = st.norm.rvs(scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        half = size // 2
        ypre[:half, -1] = st.norm.rvs(scale=np.sqrt(h[:half]))
        ypre[half:, -1] = st.norm.rvs(loc=-np.sqrt(h[half:]), scale=np.sqrt(h[half:]))
        return pars, ypre

    def proposal(self, pars, ypre):
        good = self.__test(pars)
        pars = pars[good]
        ypre = ypre[good]
        h = self.process(pars)
        pdfq = self.pdf_trunc(pars)
        for i in range(ypre.shape[1] - 1):
            pdfq *= st.norm.pdf(x=ypre[:, i], scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        pdfq *= (st.norm.pdf(x=ypre[:, -1], scale=np.sqrt(h)) +
                 st.norm.pdf(x=ypre[:, -1], loc=-np.sqrt(h), scale=np.sqrt(h))) / 2

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
            pdfp *= st.norm.pdf(x=ypre[:, i], scale=np.sqrt(h))
            h = np.exp(pars[:, 0]) + pars[:, 1] * ypre[:, i] ** 2 + pars[:, 2] * h
            h = self.__supp(h)

        out = 1.0 * np.zeros_like(good)
        out[good] = pdfp
        return out


def main():
    garch = GARCH()
    garch.laplace(inflate=2, df=1)


if __name__ == '__main__':
    main()
