import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(num):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/garch/garch' + str(num), 'rb')
    data = pickle.load(file)
    data = np.array([[da[0] for da in dat] for dat in data])
    return data


def plot(data, ax, line, name):
    x = ['1, 0.05', '1, 0.01', '2, 0.05', '2, 0.01', '5, 0.05', '5, 0.01']
    colors = ['y', 'c', 'r', 'k']
    labels = ['NIS/IS ', 'MIS/IS ', 'RIS/IS ', 'MLE/IS ']
    for i, dat in enumerate(data.T):
        ax.semilogy(x, dat, colors[i] + line, label=labels[i] + name)


def main():
    data = np.vstack([read(num) for num in np.arange(1, 31)])
    means = data.mean(axis=0)
    nvars = data.var(axis=0)
    nvars[:, 0] = 2000 * 3000 * nvars[:, 0]
    nvars[:, 1:] = 400000 * nvars[:, 1:]

    data1 = means[:, 0::2]
    print(data1)
    data2 = nvars[:, 2::2] / nvars[:, 0].reshape([-1, 1])
    data3 = means[:, 3::2] / means[:, 1].reshape([-1, 1])
    print(data2[:, 1] / data3[:, 1])
    print(data2[:, 1] / data2[:, 0])
    print(data2[:, 2] / data2[:, 1])

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    plot(data2, ax, line='-', name='nVAR')
    plot(data3, ax, line='--', name='mean(a-var)')
    ax.legend(loc=0)
    ax.set_xlabel('scenarios')
    fig.tight_layout()
    fig.show()
    return data


if __name__ == '__main__':
    R = main()
