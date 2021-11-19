import numpy as np
from matplotlib import pyplot as plt
import pickle
import sklearn.linear_model as lm


def read(dim):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/normal/time_' + str(dim) + 'D', 'rb')
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
    size_kns = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 1000])
    lines = ['-', '--', '-.', ':', '.']
    for i, dat in enumerate(data):
        ax.loglog(size_kns, dat, c + lines[i], label=labels[i])

    data = data.mean(axis=0)
    fit = lm.LinearRegression().fit(np.log(size_kns.reshape([-1, 1])), np.log(data))
    return fit.coef_[0]


def main(dim):
    times = read(dim)
    time = times.sum(axis=0)
    fig, ax = plt.subplots(figsize=[10, 8])
    result = [plot(time[:, :, 0], ax, labels=['IS1', 'IS2', 'IS3', 'IS4', 'IS5'], c='b'),
              plot(time[:, :, 1], ax, labels=['NIS1', 'NIS2', 'NIS3', 'NIS4', 'NIS5'], c='y'),
              plot(time[:, :, 2], ax, labels=['RIS1', 'RIS2', 'RIS3', 'RIS4', 'RIS5'], c='c'),
              plot(time[:, :, 3], ax, labels=['MLE1', 'MLE2', 'MLE3', 'MLE4', 'MLE5'], c='r')]
    ax.legend()
    fig.show()
    return result


if __name__ == '__main__':
    R = [main(4),
         main(6),
         main(8)]
