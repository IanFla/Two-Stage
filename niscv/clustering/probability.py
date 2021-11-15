import numpy as np
from niscv.clustering.kde2 import KDE2
from niscv.basic.expectation import Expectation

import scipy.stats as st


class Probability(Expectation):
    def __init__(self, dim, target, indicator, init_proposal, size_est, show=True):
        Expectation.__init__(self, dim, target, indicator, init_proposal, size_est, sn=True, show=show)

    def density_estimation(self, bw=1.0, mode=0, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1):
        self.kde = KDE2(self.centers, self.weights_kn, labels=self.fun(self.centers), bw=bw,
                        mode=mode, factor=factor, local=local, gamma=gamma, df=df)
        self.nonpar_proposal = self.kde.pdf
        self.nonpar_sampler = self.kde.rvs
        self.mix_proposal = lambda x: alpha0 * self.init_proposal(x) + (1 - alpha0) * self.nonpar_proposal(x)
        self.mix_sampler = lambda size: np.vstack([self.init_sampler(round(alpha0 * size)),
                                                   self.nonpar_sampler(size - round(alpha0 * size))])
        self.controls = lambda x: self.kde.kernels(x) - self.mix_proposal(x)


def experiment(dim, b, size_est, show, size_kn, ratio, mode=0):
    results = []
    mean = np.zeros(dim)
    target = lambda x: st.multivariate_normal(mean=mean).pdf(x)
    indicator = lambda x: 1 * (x[:, 0] > b)
    init_proposal = st.multivariate_normal(mean=mean, cov=4)
    grid_x = np.linspace(-5, 5, 200)
    pro = Probability(dim, target, indicator, init_proposal, size_est, show=show)
    pro.initial_estimation(size_kn, ratio, resample=True)
    results.extend([pro.result[-5], pro.result[-4]])
    if pro.show:
        pro.draw(grid_x, name='initial')

    pro.density_estimation(bw=1.0, mode=mode, factor='scott', local=False, gamma=0.3, df=0, alpha0=0.1)
    pro.nonparametric_estimation()
    results.extend([pro.result[-4], pro.result[-3], pro.result[-2], pro.result[-1]])
    if pro.show:
        pro.draw(grid_x, name='nonparametric')

    pro.regression_estimation()
    results.extend([pro.result[-2], pro.result[-1]])
    if pro.show:
        pro.draw(grid_x, name='regression')

    pro.likelihood_estimation(optimize=True, NR=True)
    results.extend([pro.result[-1], results[-1]])
    return results


def main():
    np.random.seed(3033079628)
    results = []
    for i in range(100):
        print(i + 1)
        result = experiment(dim=4, b=2, size_est=10000, show=False, size_kn=300, ratio=20, mode=0)
        results.append(result)

    return np.array(results)


if __name__ == '__main__':
    truth = st.norm.cdf(-2)
    R = main()

    aVar = R[:, 1::2]
    mean_aVar = aVar.mean(axis=0)
    std_aVar = aVar.std(axis=0)
    print('a-var(l):', np.round(mean_aVar - 1.96 * std_aVar, 4))
    print('a-var(m):', np.round(mean_aVar, 4))
    print('a-var(r):', np.round(mean_aVar + 1.96 * std_aVar, 4))

    MSE = np.mean((R[:, ::2] - truth) ** 2, axis=0)
    print('nMSE:', np.round(np.append(6000 * MSE[0], 10000 * MSE[1:]), 4))

    aErr = np.sqrt(np.hstack([aVar[:, 0].reshape([-1, 1]) / 6000, aVar[:, 1:] / 10000]))
    Flag = (truth >= R[:, ::2] - 1.96 * aErr) & (truth <= R[:, ::2] + 1.96 * aErr)
    print('C.I.:', Flag.mean(axis=0))
