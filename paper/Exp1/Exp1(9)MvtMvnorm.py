import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvnorm
from scipy.stats import multivariate_t as mvt


class TN:
    def __init__(self, dim):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.norm = mvnorm(mean=self.mean)

    def draw(self):
        Df = [3, 4, 5, 8, 12, 20, 50, 100]
        x = np.linspace(-4, 4, 1000)
        X = np.zeros([1000, self.dim])
        X[:, 0] = x
        fig, ax = plt.subplots()
        ax.plot(x, self.norm.pdf(x=X), label='norm')
        for df in Df:
            t = mvt(loc=self.mean, shape=(df-2)/df, df=df)
            ax.plot(x, t.pdf(x=X), label='t('+str(df)+')')

        ax.legend()
        ax.set_title(str(self.dim)+'-D')
        plt.show()


def main():
    for dim in np.arange(2, 11):
        TN(dim).draw()


if __name__ == '__main__':
    main()
