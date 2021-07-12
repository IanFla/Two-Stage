import numpy as np


if __name__ == '__main__':
    size = 10000
    W = np.abs(np.random.exponential(1, size))
    W = W / W.sum()
    neff = 1 / (W ** 2).sum()
    W = W / W.max()
    nacc = W.sum()
    print('n: {:.0f}, neff: {:.0f}, nacc: {:.0f}'.format(size, neff, nacc))
