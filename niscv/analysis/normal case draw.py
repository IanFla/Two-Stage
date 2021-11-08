import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st


def read(dim):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/normal5_' + str(dim) + 'D', 'rb')
    data = pickle.load(file)
    data = np.array([da[0] for da in data])
    return data


def plot(data, ax, label, c, mode='a-var', truth=None, n=''):
    size_kns = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    if mode == 'nmse':
        n = 500 * size_kns if n == 'IS' else 5000
        nMSE = n * np.mean((data - truth) ** 2, axis=0)
        ax.loglog(size_kns, nMSE, c, label=label)
    elif mode == 'nvar':
        n = 500 * size_kns if n == 'IS' else 5000
        nvar = n * np.var(data, axis=0)
        ax.loglog(size_kns, nvar, c + '.', label=label)
    elif mode == 'a-var':
        avar_mean = data.mean(axis=0)
        ax.loglog(size_kns, avar_mean, c + '--', label=label)
    else:
        print('mode error! ')


def draw(dim, order, sn, ax):
    data = read(dim)
    index = 0 if order == 0 else 2 * order - 1 + sn
    name = str(dim) + 'D, M' + str(order) + ', SN(' + str(sn) + ')'
    truth = st.norm.moment(order) + 1

    plot(data[:, index, :, 0], ax, label='IS nMSE', c='b', mode='nmse', truth=truth, n='IS')
    plot(data[:, index, :, 0], ax, label='IS nVAR', c='b', mode='nvar', truth=truth, n='IS')
    plot(data[:, index, :, 1], ax, label='IS mean(a-var)', c='b', mode='a-var', truth=None)
    plot(data[:, index, :, 2], ax, label='NIS nMSE', c='y', mode='nmse', truth=truth)
    plot(data[:, index, :, 2], ax, label='NIS nVAR', c='y', mode='nvar', truth=truth)
    plot(data[:, index, :, 3], ax, label='NIS mean(a-var)', c='y', mode='a-var', truth=None)
    plot(data[:, index, :, 4], ax, label='MIS nMSE', c='c', mode='nmse', truth=truth)
    plot(data[:, index, :, 4], ax, label='MIS nVAR', c='c', mode='nvar', truth=truth)
    plot(data[:, index, :, 5], ax, label='MIS mean(a-var)', c='c', mode='a-var', truth=None)
    plot(data[:, index, :, 6], ax, label='RIS nMSE', c='r', mode='nmse', truth=truth)
    plot(data[:, index, :, 6], ax, label='RIS nVAR', c='r', mode='nvar', truth=truth)
    plot(data[:, index, :, 7], ax, label='RIS mean(a-var)', c='r', mode='a-var', truth=None)
    plot(data[:, index, :, 8], ax, label='MLE nMSE', c='k', mode='nmse', truth=truth)
    plot(data[:, index, :, 8], ax, label='MLE nVAR', c='k', mode='nvar', truth=truth)
    # ax.set_xlabel('log(kernel number)')
    ax.legend()
    ax.set_title(name)


def main(dim, ax):
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True], [3, False], [3, True], [4, False], [4, True]]
    for j, setting in enumerate(settings):
        draw(dim, order=setting[0], sn=setting[1], ax=ax[j])


if __name__ == '__main__':
    plt.style.use('ggplot')
    dims = [5, 7, 9]
    fig, axs = plt.subplots(9, 3, figsize=[30, 50])
    for i, d in enumerate(dims):
        main(d, axs[:, i])

    fig.show()
