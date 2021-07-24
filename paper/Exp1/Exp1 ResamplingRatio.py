import numpy as np
from scipy.stats import multivariate_normal as mvnorm


class RRR:
    def __init__(self, dim, sigma1, sigma2):
        self.dim = dim
        mean = np.zeros(dim)
        self.target = mvnorm(mean=mean, cov=sigma1 ** 2)
        self.init_proposal = mvnorm(mean=mean, cov=sigma2 ** 2)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def calculate(self, size):
        the_n_ESS = ((self.sigma2 ** 2 / self.sigma1) ** self.dim) \
                  / (2 * self.sigma2 ** 2 - self.sigma1 ** 2) ** (self.dim / 2)
        the_n_RSS = (self.sigma2 / self.sigma1) ** self.dim

        samples = self.init_proposal.rvs(size=size)
        weights = self.target.pdf(x=samples) / self.init_proposal.pdf(x=samples)
        cal_ESS = 1 / ((weights / weights.sum()) ** 2).sum()
        cal_RSS = weights.sum() / weights.max()

        print('{}D: n/ESS(Cal/The): {:.0f}/{:.0f}; n/RSS(Cal/The): {:.0f}/{:.0f}'
              .format(self.dim, size / cal_ESS, the_n_ESS, size / cal_RSS, the_n_RSS))


def main():
    for dim in range(10):
        RRR(dim + 1, 1, 2).calculate(100000)


if __name__ == '__main__':
    main()
