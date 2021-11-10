import numpy as np
from matplotlib import pyplot as plt
import pickle
import scipy.stats as st
import sklearn.linear_model as lm
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def read(dim):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/normal/convergence_' + str(dim) + 'D', 'rb')
    data = pickle.load(file)
    data = [da[0] for da in data]
    return np.array(data)


def plot(data, ax, label, c, mode, truth):
    size_kns = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    if mode == 'nmse':
        MSE = np.mean((data - truth) ** 2, axis=0).astype(np.float64)
        ax.loglog(size_kns, MSE, c, label=label)
        fit = lm.LinearRegression().fit(np.log(size_kns.reshape([-1, 1])), np.log(MSE))
        return fit.coef_[0]
    elif mode == 'nvar':
        var = np.var(data, axis=0)
        ax.loglog(size_kns, var, c + '.', label=label)
    else:
        print('mode error! ')


def draw(dim, order, sn, ax):
    data = read(dim)
    index = 0 if order == 0 else 2 * order - 1 + sn
    name = str(dim) + 'D, M' + str(order) + ', SN(' + str(sn) + ')'
    truth = st.norm.moment(order) + 1

    result = []
    plot(data[:, index, :, 1], ax, label='NIS nMSE', c='y', mode='nmse', truth=truth)
    plot(data[:, index, :, 1], ax, label='NIS nVAR', c='y', mode='nvar', truth=truth)
    result.append(plot(data[:, index, :, 2], ax, label='MIS nMSE', c='c', mode='nmse', truth=truth))
    plot(data[:, index, :, 2], ax, label='MIS nVAR', c='c', mode='nvar', truth=truth)
    plot(data[:, index, :, 3], ax, label='RIS nMSE', c='r', mode='nmse', truth=truth)
    plot(data[:, index, :, 3], ax, label='RIS nVAR', c='r', mode='nvar', truth=truth)
    result.append(plot(data[:, index, :, 4], ax, label='MLE nMSE', c='k', mode='nmse', truth=truth))
    plot(data[:, index, :, 4], ax, label='MLE nVAR', c='k', mode='nvar', truth=truth)
    ax.set_xlabel('log(kernel number)')
    ax.legend(loc=3)
    ax.set_title(name)

    return result


def main(dim, ax):
    settings = [[0, False], [1, False], [1, True], [2, False], [2, True]]
    result = []
    for j, setting in enumerate(settings):
        result.append(draw(dim, order=setting[0], sn=setting[1], ax=ax[j]))

    return result


if __name__ == '__main__':
    plt.style.use('ggplot')
    dims = [4, 6, 8]
    R = []
    fig, axs = plt.subplots(5, 3, figsize=[30, 30])
    for i, d in enumerate(dims):
        R.append(main(d, axs[:, i]))

    fig.tight_layout()
    fig.show()
