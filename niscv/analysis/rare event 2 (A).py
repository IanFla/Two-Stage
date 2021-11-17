import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st


def read(b):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/rare/rare4_(' + str(b) + ')', 'rb')
    data = pickle.load(file)
    data = np.array([da[0] for da in data])
    return data


def plot(data, ax, label, c, mode='a-var', truth=None, control=''):
    modes = ['none', 'auto', 'km(2)', 'km(3)', 'km(4)', 'km(5)']
    n = 500 * 1000 if control == 'IS' else 10000
    if mode == 'nmse':
        nMSE = n * np.mean((data - truth) ** 2, axis=0)
        ax.semilogy(modes, nMSE, c, label=label)
        print(nMSE)
    elif mode == 'nvar':
        nvar = n * np.var(data, axis=0)
        ax.semilogy(modes, nvar, c + '.', label=label)
    elif mode == 'a-var':
        avar_mean = data.mean(axis=0)
        avar_mean = avar_mean * 10000 / (10000 - 500 - 1) if control == 'RIS' else avar_mean
        ax.semilogy(modes, avar_mean, c + '--', label=label)
    else:
        print('mode error! ')


def draw(data, truth, ax):
    plot(data[:, :, 0], ax, label='IS nMSE', c='b', mode='nmse', truth=truth, control='IS')
    plot(data[:, :, 0], ax, label='IS nVAR', c='b', mode='nvar', truth=truth, control='IS')
    plot(data[:, :, 1], ax, label='IS mean(a-var)', c='b', mode='a-var')
    plot(data[:, :, 2], ax, label='NIS nMSE', c='y', mode='nmse', truth=truth)
    plot(data[:, :, 2], ax, label='NIS nVAR', c='y', mode='nvar', truth=truth)
    plot(data[:, :, 3], ax, label='NIS mean(a-var)', c='y', mode='a-var')
    plot(data[:, :, 4], ax, label='MIS nMSE', c='c', mode='nmse', truth=truth)
    plot(data[:, :, 4], ax, label='MIS nVAR', c='c', mode='nvar', truth=truth)
    plot(data[:, :, 5], ax, label='MIS mean(a-var)', c='c', mode='a-var')
    plot(data[:, :, 6], ax, label='RIS nMSE', c='r', mode='nmse', truth=truth)
    plot(data[:, :, 6], ax, label='RIS nVAR', c='r', mode='nvar', truth=truth)
    plot(data[:, :, 7], ax, label='RIS mean(a-var)', c='r', mode='a-var', control='RIS')
    plot(data[:, :, 8], ax, label='MLE nMSE', c='k', mode='nmse', truth=truth)
    plot(data[:, :, 8], ax, label='MLE nVAR', c='k', mode='nvar', truth=truth)
    ax.set_xlabel('clustering state')


def main(b, ax):
    data = read(b)
    truth = st.norm.cdf(-b)
    print('reference:', truth * (1 - truth), 4 * (truth * (1 - truth)) ** 2)
    draw(data=data, truth=truth, ax=ax)
    modes = ['none', 'auto', 'km(2)', 'km(3)', 'km(4)', 'km(5)']
    ax.semilogy(modes, truth * (1 - truth) * np.ones(6), '-.', color='orange', label='reference 1')
    ax.semilogy(modes, 4 * (truth * (1 - truth)) ** 2 * np.ones(6), 'm-.', label='reference 2')
    ax.legend(loc=1)
    ax.set_title('b(' + str(b) + ')')


if __name__ == '__main__':
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 1, figsize=[8, 6])
    main(b=2, ax=axs)
    fig.tight_layout()
    fig.show()
