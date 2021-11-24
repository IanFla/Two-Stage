import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/garch/garch_bw2', 'rb')
    data = pickle.load(file)
    data = np.array([[da[0] for da in dat] for dat in data])
    return data


def plot(data, ax, name):
    BW = [0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6]
    colors = ['b', 'y', 'c', 'r']
    lines = ['-', '--']
    labels = [['IS (0.05)', 'IS (0.01)'], ['NIS (0.05)', 'NIS (0.01)'],
              ['MIS (0.05)', 'MIS (0.01)'], ['RIS (0.05)', 'RIS (0.01)']]
    for i, color in enumerate(colors):
        for j, line in enumerate(lines):
            ax.semilogy(BW, data[:, j, i], color + line, label=labels[i][j])

    ax.legend(loc=1)
    ax.set_xlabel('bandwidth')
    ax.set_ylabel('log(a-var)')
    ax.set_title(name)


def main():
    data = read()
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 3, figsize=[25, 7])
    plot(data[:, 0:2, [-7, -5, -3, -1]], ax[0], '4D (d=1)')
    plot(data[:, 2:4, [-7, -5, -3, -1]], ax[1], '5D (d=2)')
    plot(data[:, 4:6, [-7, -5, -3, -1]], ax[2], '8D (d=5)')
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
