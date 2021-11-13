import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy.spatial.distance import mahalanobis
from particles import resampling as rs


class KDE:
    def __init__(self, centers, weights, bw=1.0, factor='scott', local=False, gamma=0.3, df=0):
        self.centers = centers
        self.weights = weights / weights.sum()
        self.ESS = 1 / np.sum(self.weights ** 2)
        self.n, self.d = centers.shape
        self.local = local
        if self.local:
            icov = np.linalg.inv(np.cov(self.centers.T, aweights=weights))
            distances = []
            for x1 in self.centers:
                dists = []
                for x2 in self.centers:
                    dists.append(mahalanobis(x1, x2, icov))

                distances.append(dists)

            covs = []
            for j, center in enumerate(self.centers):
                index = np.argsort(distances[j])[:np.around(gamma * self.n).astype(np.int64)]
                covs.append(np.cov(self.centers[index].T, aweights=weights[index]))

        else:
            covs = np.cov(centers.T, aweights=weights)

        scott = self.ESS ** (-1 / (self.d + 4))
        silverman = scott * ((4 / (self.d + 2)) ** (1 / (self.d + 4)))
        self.factor = bw * scott if factor == 'scott' else bw * silverman
        self.factor = self.factor / (gamma ** (1 / self.d)) if self.local else self.factor
        self.covs = (self.factor ** 2) * np.array(covs)
        if df == 0:
            self.kernel_pdf = lambda x, m, v: st.multivariate_normal.pdf(x=x, mean=m, cov=v)
            self.kernel_rvs = lambda size, m, v: st.multivariate_normal.rvs(size=size, mean=m, cov=v)
        elif df >= 3:
            self.kernel_pdf = lambda x, m, v: st.multivariate_t.pdf(x=x, loc=m, shape=(df - 2) * v / df, df=df)
            self.kernel_rvs = lambda size, m, v: st.multivariate_t.rvs(size=size, loc=m, shape=(df - 2) * v / df, df=df)
        else:
            print('df err! ')

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.covs
            density += self.weights[j] * self.kernel_pdf(x=samples, m=center, v=cov)

        return density

    def rvs(self, size):
        index, sizes = np.unique(rs.stratified(self.weights, M=size), return_counts=True)
        cum_sizes = np.append(0, np.cumsum(sizes))
        samples = np.zeros([size, self.d])
        for j, center in enumerate(self.centers[index]):
            cov = self.covs[j] if self.local else self.covs
            samples[cum_sizes[j]:cum_sizes[j + 1]] = self.kernel_rvs(size=sizes[j], m=center, v=cov)

        return samples

    def kernels(self, x):
        out = np.zeros([self.centers.shape[0], x.shape[0]])
        for j, center in enumerate(self.centers):
            cov = self.covs[j] if self.local else self.covs
            out[j] = self.kernel_pdf(x=x, m=center, v=cov)

        return out


def main(bw, factor, local, gamma, df, seed=19971107):
    np.random.seed(seed)
    target = lambda x: 0.7 * st.multivariate_normal(mean=[-1, 0], cov=[8, 0.2]).pdf(x) + \
                       0.3 * st.multivariate_normal(mean=[1, 0], cov=[0.25, 4]).pdf(x)
    proposal = st.multivariate_normal(mean=[-1, 0], cov=4).pdf
    centers = st.multivariate_normal(mean=[-1, 0], cov=4).rvs(size=1000)
    weights = target(centers) / proposal(centers)

    kde = KDE(centers=centers, weights=weights, bw=bw, factor=factor, local=local, gamma=gamma, df=df)
    samples = kde.rvs(size=1000)

    grid_x = np.linspace(-7, 5, 200)
    grid_y = np.linspace(-4, 4, 200)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_proposal = proposal(grids).reshape(grid_X.shape)
    grid_Z_kde = kde.pdf(grids).reshape(grid_X.shape)

    fig, AX = plt.subplots(2, 2, figsize=[15, 10])
    AX[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    AX[0, 1].contour(grid_X, grid_Y, grid_Z_kde)
    AX[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    AX[1, 0].scatter(centers[:, 0], centers[:, 1])
    AX[1, 1].contour(grid_X, grid_Y, grid_Z_kde)
    AX[1, 1].scatter(samples[:, 0], samples[:, 1])
    for ax in AX.flatten():
        ax.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        ax.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    plt.show()


if __name__ == '__main__':
    main(bw=1.0, factor='scott', local=True, gamma=0.3, df=0)
