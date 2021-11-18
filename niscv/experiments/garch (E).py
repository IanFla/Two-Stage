import numpy as np
from matplotlib import pyplot as plt
from niscv.clustering.quantile import Quantile
from niscv.real.garch import GARCH
import pandas as pd
import seaborn as sb


class IP:
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs


def draw(self):
    df = pd.DataFrame(self.centers, columns=['phi0', 'phi1', 'beta'] +
                                            ['y{}'.format(i + 1) for i in range(self.centers.shape[1] - 3)])
    df['cluster'] = self.kde.labels + 1
    sb.pairplot(df, hue='cluster')
    plt.show()


def experiment(d, alpha, size_est, show, size_kn, ratio, bw, km, local, gamma, alpha0, server=True):
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
    qtl.density_estimation(bw=bw, km=km, factor='scott', local=local, gamma=gamma, df=0, alpha0=alpha0)
    draw(qtl)
    qtl.nonparametric_estimation()
    results.extend([qtl.result[-4], qtl.result[-3], qtl.result[-2], qtl.result[-1]])
    qtl.regression_estimation()
    results.extend([qtl.result[-2], qtl.result[-1]])
    qtl.likelihood_estimation(optimize=True, NR=True)
    results.append(qtl.result[-1])
    return results, qtl.result


def main():
    experiment(d=2, alpha=0.05, size_est=100000, show=True,
               size_kn=2000, ratio=1000, bw=1.3, km=2, local=True, gamma=0.3, alpha0=0.1, server=True)


if __name__ == '__main__':
    main()
