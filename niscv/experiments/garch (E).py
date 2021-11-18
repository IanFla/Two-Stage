import numpy as np
# from matplotlib import pyplot as plt
from niscv.clustering.quantile import Quantile
from niscv.real.garch import GARCH


class IP:
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs


def experiment(d, alpha, size_est, show, size_kn, ratio, bw, mode, local, gamma, alpha0, server=True):
    results = []
    garch = GARCH(server=server)
    garch.laplace(inflate=2, df=1)

    target = lambda x: garch.target(x[:, :3], x[:, 3:])
    statistic = lambda x: x[:, 3:].sum(axis=1)
    init_proposal = IP(pdf=lambda x: garch.proposal(x[:, :3], x[:, 3:]),
                       rvs=lambda size: np.hstack(garch.predict(d, size)))
    qtl = Quantile(3 + d, target, statistic, alpha, init_proposal, size_est, show=show)
    qtl.initial_estimation(size_kn, ratio)
    results.extend([qtl.result[-5], qtl.result[-4]])
    qtl.density_estimation(bw=bw, mode=mode, factor='scott', local=local, gamma=gamma, df=0, alpha0=alpha0)
    qtl.nonparametric_estimation()
    results.extend([qtl.result[-4], qtl.result[-3], qtl.result[-2], qtl.result[-1]])
    qtl.regression_estimation()
    results.extend([qtl.result[-2], qtl.result[-1]])
    qtl.likelihood_estimation(optimize=True, NR=True)
    results.append(qtl.result[-1])
    return results, qtl.result


def main():
    experiment(d=2, alpha=0.05, size_est=100000, show=True,
               size_kn=1000, ratio=1000, bw=1.0, mode=0, local=True, gamma=0.3, alpha0=0.2, server=False)


if __name__ == '__main__':
    main()
