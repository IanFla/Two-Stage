import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn.linear_model as lm


def read(dim):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/normal/time2_' + str(dim) + 'D', 'rb')
    Data = pickle.load(file)
    Results = []
    for data in Data:
        results = []
        for dat in data:
            result = []
            for da in dat:
                res = []
                for d in da:
                    res.append(d.seconds + d.microseconds / 1000000)

                result.append(res)

            results.append(result)

        Results.append(results)

    return np.array(Results)


def plot(data, ax, labels, c):
    # size_kns = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 1000])
    size_ests = np.array([1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 70000, 100000])
    lines = ['-', '--', '-.', ':', '.']
    for i, dat in enumerate(data):
        ax.plot(size_ests, dat, c + lines[i], label=labels[i])

    data = data.mean(axis=0)
    fit = lm.LinearRegression().fit(np.log(size_ests.reshape([-1, 1])), np.log(data))
    return fit.coef_[0]


def main(dim, ax, name):
    times = read(dim)
    time = times.sum(axis=0)
    result = [plot(time[:, :, 0], ax, labels=['IS+re 1', 'IS+re 1', 'IS+re 1', 'IS+re 1', 'IS+re 1'], c='b'),
              plot(time[:, :, 1], ax, labels=['NIS+MIS 1', 'NIS+MIS 2', 'NIS+MIS 3', 'NIS+MIS 4', 'NIS+MIS 5'], c='y'),
              plot(time[:, :, 2], ax, labels=['RIS 1', 'RIS 2', 'RIS 3', 'RIS 4', 'RIS 5'], c='r'),
              plot(time[:, :, 3], ax, labels=['MLE 1', 'MLE 2', 'MLE 3', 'MLE 4', 'MLE 5'], c='k')]
    ax.legend(loc=1)
    ax.set_xlabel('sample size')
    ax.set_ylabel('time - seconds')
    ax.set_ylim([0, 100])
    ax.set_title(name)
    return result


if __name__ == '__main__':
    plt.style.use('ggplot')
    dims = [4, 6, 8]
    R = []
    fig, axs = plt.subplots(1, 3, figsize=[25, 7])
    for j, di in enumerate(dims):
        R.append(main(di, axs[j], str(di)+'D'))

    fig.tight_layout()
    fig.show()
