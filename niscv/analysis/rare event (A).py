import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st


def read(b):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/rare/rare_(' + str(b) + ')', 'rb')
    data = pickle.load(file)
    data1 = np.array([da[0][0] for da in data])
    data2 = np.array([da[0][1:] for da in data])
    return data1, data2


def plot(data, ax, label, c, mode='a-var', truth=None, n=''):
    ratios = np.array([5, 10, 15, 20, 30, 50, 70, 100, 150, 200, 500, 1000])
    if mode == 'nmse':
        n = 500 * ratios if n == 'IS' else 10000
        nMSE = n * np.mean((data - truth) ** 2, axis=0)
        ax.loglog(ratios, nMSE, c, label=label)
    elif mode == 'nvar':
        n = 500 * ratios if n == 'IS' else 10000
        nvar = n * np.var(data, axis=0)
        ax.loglog(ratios, nvar, c + '.', label=label)
    elif mode == 'a-var':
        avar_mean = data.mean(axis=0)
        avar_mean = avar_mean * 10000 / (10000 - 500 - 1) if n == 'RIS' else avar_mean
        ax.loglog(ratios, avar_mean, c + '--', label=label)
    else:
        print('mode error! ')


def draw(data, name, truth, ax):
    plot(data[:, :, 0], ax, label='IS nMSE', c='b', mode='nmse', truth=truth, n='IS')
    plot(data[:, :, 0], ax, label='IS nVAR', c='b', mode='nvar', truth=truth, n='IS')
    plot(data[:, :, 1], ax, label='IS mean(a-var)', c='b', mode='a-var', truth=None)
    plot(data[:, :, 2], ax, label='NIS nMSE', c='y', mode='nmse', truth=truth)
    plot(data[:, :, 2], ax, label='NIS nVAR', c='y', mode='nvar', truth=truth)
    plot(data[:, :, 3], ax, label='NIS mean(a-var)', c='y', mode='a-var', truth=None)
    plot(data[:, :, 4], ax, label='MIS nMSE', c='c', mode='nmse', truth=truth)
    plot(data[:, :, 4], ax, label='MIS nVAR', c='c', mode='nvar', truth=truth)
    plot(data[:, :, 5], ax, label='MIS mean(a-var)', c='c', mode='a-var', truth=None)
    plot(data[:, :, 6], ax, label='RIS nMSE', c='r', mode='nmse', truth=truth)
    plot(data[:, :, 6], ax, label='RIS nVAR', c='r', mode='nvar', truth=truth)
    plot(data[:, :, 7], ax, label='RIS mean(a-var)', c='r', mode='a-var', truth=None, n='RIS')
    plot(data[:, :, 8], ax, label='MLE nMSE', c='k', mode='nmse', truth=truth)
    plot(data[:, :, 8], ax, label='MLE nVAR', c='k', mode='nvar', truth=truth)
    ax.set_xlabel('log(kernel number)')
    ax.legend(loc=3)
    ax.set_title(name)


def main(b, ax):
    data1, data2 = read(b)
    truth = st.norm.cdf(-b)
    settings = [[True, 1], [False, 2], [True, 2], [True, 3], [True, 4]]
    for j, setting in enumerate(settings):
        name = '5D, K' + str(setting[1]) + ', auto(' + str(setting[0]) + ')'
        draw(data=data2[:, j, :, :], name=name, truth=truth, ax=ax[j])


if __name__ == '__main__':
    plt.style.use('ggplot')
    fig, axs = plt.subplots(5, 1, figsize=[10, 30])
    main(b=2.0, ax=axs)
    fig.tight_layout()
    fig.show()
