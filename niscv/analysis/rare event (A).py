import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st


def read(b):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/rare/rare3_(' + str(b) + ')', 'rb')
    data = pickle.load(file)
    data1 = np.array([da[0][0] for da in data])
    data2 = np.array([da[0][1:] for da in data])
    data21 = data2[:, :, [0, 1, 6, 7, 8, 9, 10, 11, 12]]
    data22 = data2[:, :, [2, 4]]
    return data1, data21, data22


def plot(data, ax, label, c, mode='a-var', truth=None, control=''):
    ratios = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 200, 500, 1000])
    n = 500 * ratios if control == 'IS' else 10000
    if mode == 'nmse':
        nMSE = n * np.mean((data - truth) ** 2, axis=0)
        ax.loglog(ratios, nMSE, c, label=label)
    elif mode == 'nvar':
        nvar = n * np.var(data, axis=0)
        ax.loglog(ratios, nvar, c + '.', label=label)
    elif mode == 'a-var':
        avar_mean = data.mean(axis=0)
        avar_mean = avar_mean * 10000 / (10000 - 500 - 1) if control == 'RIS' else avar_mean
        ax.loglog(ratios, avar_mean, c + '--', label=label)
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
    ax.set_xlabel('log(resampling ratios)')


def main(b, ax):
    data1, data21, data22 = read(b)
    truth = st.norm.cdf(-b)
    draw(data=data21, truth=truth, ax=ax)
    ax.legend(loc=1)
    ax.set_title('b(' + str(b) + ')')

    SS_mean = data22.mean(axis=0)
    n0_ESS = 1000 * 500 / SS_mean[-1, 0]
    m_m0 = SS_mean[:, 1] / 500
    ratios = np.array([2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 200, 500, 1000])
    DSS09 = ratios[m_m0 > 0.9][0]

    nMSE1 = 10000 * np.mean((data1[:, 4::2] - truth) ** 2, axis=0)
    nMSE2 = 10000 * np.mean((data21[:, 0, 4:-1:2] - truth) ** 2, axis=0)

    return 2 * n0_ESS, DSS09, nMSE1, nMSE2


if __name__ == '__main__':
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=[20, 15])
    axs = axs.flatten()
    R = []
    for i in range(4):
        R.append(main(b=i, ax=axs[i]))

    fig.tight_layout()
    fig.show()
