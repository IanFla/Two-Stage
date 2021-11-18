import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from particles import resampling as rs
from niscv.basic.kde import KDE


class KDE2:
    def __init__(self, centers, weights, bw, mode=1, factor='scott', local=False, gamma=0.3, df=0, labels=None):
        if mode == 1:
            labels = np.zeros(centers.shape[0]).astype(np.int32)
        elif mode > 1:
            scaler = StandardScaler().fit(centers, sample_weight=weights)
            kmeans = KMeans(n_clusters=mode).fit(scaler.transform(centers), sample_weight=weights)
            labels = kmeans.labels_

        self.labels = labels
        nums = np.array([weights[labels == i].sum() for i in range(labels.max(initial=0) + 1)])
        self.prop = nums / nums.sum()

        self.kdes = []
        for i in range(labels.max(initial=0) + 1):
            label = (labels == i)
            self.kdes.append(KDE(centers[label], weights[label], bw, factor, local, gamma, df))

    def pdf(self, samples):
        density = np.zeros(samples.shape[0])
        for l, kde in enumerate(self.kdes):
            density += self.prop[l] * kde.pdf(samples)

        return density

    def rvs(self, size):
        sizes = np.unique(rs.stratified(self.prop, M=size), return_counts=True)[1]
        return np.vstack([kde.rvs(sz) for kde, sz in zip(self.kdes, sizes)])

    def kernels(self, x):
        return np.vstack([kde.kernels(x) for kde in self.kdes])


def main(bw, factor, local, gamma, df, mode, seed=19971107):
    np.random.seed(seed)
    target = lambda x: 0.7 * st.multivariate_normal(mean=[-1, 0], cov=0.4).pdf(x) + \
                       0.3 * st.multivariate_normal(mean=[2, 0], cov=0.2).pdf(x)
    proposal = st.multivariate_normal(mean=[0, 0], cov=4).pdf
    centers = st.multivariate_normal(mean=[0, 0], cov=4).rvs(size=1000)
    weights = target(centers) / proposal(centers)

    kde1 = KDE(centers, weights, bw, factor=factor, local=local, gamma=gamma, df=df)
    samples1 = kde1.rvs(size=2000)
    kde2 = KDE2(centers, weights, bw, mode=mode, factor=factor, local=local, gamma=gamma, df=df)
    samples2 = kde2.rvs(size=2000)

    grid_x = np.linspace(-4, 4, 200)
    grid_y = np.linspace(-2, 2, 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grids = np.array([grid_X.flatten(), grid_Y.flatten()]).T
    grid_Z_target = target(grids).reshape(grid_X.shape)
    grid_Z_proposal = proposal(grids).reshape(grid_X.shape)
    grid_Z_kde1 = kde1.pdf(grids).reshape(grid_X.shape)
    grid_Z_kde2 = kde2.pdf(grids).reshape(grid_X.shape)

    fig, AX = plt.subplots(2, 2, figsize=[15, 10])
    AX[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    AX[0, 1].contour(grid_X, grid_Y, grid_Z_kde1)
    AX[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    AX[1, 0].scatter(centers[:, 0], centers[:, 1])
    AX[1, 1].contour(grid_X, grid_Y, grid_Z_kde1)
    AX[1, 1].scatter(samples1[:, 0], samples1[:, 1])
    for ax in AX.flatten():
        ax.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        ax.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()

    fig, AX = plt.subplots(2, 2, figsize=[15, 10])
    AX[0, 0].contour(grid_X, grid_Y, grid_Z_target)
    AX[0, 1].contour(grid_X, grid_Y, grid_Z_kde2)
    AX[1, 0].contour(grid_X, grid_Y, grid_Z_proposal)
    AX[1, 0].scatter(centers[:, 0], centers[:, 1])
    AX[1, 1].contour(grid_X, grid_Y, grid_Z_kde2)
    AX[1, 1].scatter(samples2[:, 0], samples2[:, 1])
    for ax in AX.flatten():
        ax.set_xlim(grid_x.min(initial=0), grid_x.max(initial=0))
        ax.set_ylim(grid_y.min(initial=0), grid_y.max(initial=0))

    fig.show()

    print(kde1.kernels(samples1).shape)
    print(kde2.kernels(samples1).shape)


if __name__ == '__main__':
    main(bw=1.0, factor='scott', local=False, gamma=0.3, df=0, mode=3)
