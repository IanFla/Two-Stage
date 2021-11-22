import numpy as np
from matplotlib import pyplot as plt
from niscv.clustering.quantile import Quantile
from niscv.real.garch import GARCH
import pandas as pd
import seaborn as sb
import multiprocessing
import os
from functools import partial
from datetime import datetime as dt
import pickle


class IP:
    def __init__(self, pdf, rvs):
        self.pdf = pdf
        self.rvs = rvs


def draw(self):
    df = pd.DataFrame(self.centers, columns=['log(phi0)', 'phi1', 'beta'] +
                                            ['y{}'.format(i + 1) for i in range(self.centers.shape[1] - 3)])
    df['cluster'] = self.kde.labels + 1
    plt.style.use('ggplot')
    sb.pairplot(df, hue='cluster')
    plt.show()


def experiment(d, alpha, size_est, show, size_kn, ratio, bw, km, local, gamma, alpha0):
    results = []
    garch = GARCH()
    garch.laplace(inflate=2, df=1)

    target = lambda x: garch.target(x[:, :3], x[:, 3:])
    statistic = lambda x: x[:, 3:].sum(axis=1)
    init_proposal = IP(pdf=lambda x: garch.proposal(x[:, :3], x[:, 3:]),
                       rvs=lambda size: np.hstack(garch.predict(d, size)))
    qtl = Quantile(3 + d, target, statistic, alpha, init_proposal, size_est, show=show)
    qtl.initial_estimation(size_kn, ratio)
    results.extend([qtl.result[-5], qtl.result[-4]])
    qtl.density_estimation(bw=bw, km=km, factor='scott', local=local, gamma=gamma, df=0, alpha0=alpha0)
    if qtl.show:
        draw(qtl)

    qtl.nonparametric_estimation()
    results.extend([qtl.result[-4], qtl.result[-3], qtl.result[-2], qtl.result[-1]])
    qtl.regression_estimation()
    results.extend([qtl.result[-2], qtl.result[-1]])
    qtl.likelihood_estimation(optimize=True, NR=True)
    results.append(qtl.result[-1])
    return results, qtl.result


def run(it, num):
    np.random.seed(1997 * num + 1107 + it)
    D = [1, 2, 5]
    Alpha = [0.05, 0.01]
    result = []
    for i, d in enumerate(D):
        for alpha in Alpha:
            print(num, it, d, alpha)
            result.append(experiment(d=d, alpha=alpha, size_est=400000, show=False, size_kn=2000, ratio=3000,
                                     bw=1.4, km=2, local=True, gamma=0.3, alpha0=0.1))

    return result


def main(num):
    os.environ['OMP_NUM_THREADS'] = '3'
    with multiprocessing.Pool(processes=10) as pool:
        begin = dt.now()
        its = np.arange(1, 11)
        R = pool.map(partial(run, num=num), its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/garch/garch' + str(num), 'wb') as file:
        pickle.dump(R, file)


if __name__ == '__main__':
    for n in np.arange(3, 31):
        main(n)
