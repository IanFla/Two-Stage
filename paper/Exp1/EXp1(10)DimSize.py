import numpy as np
from matplotlib import pyplot as plt
import pickle

file = open('Data/DimSize', 'rb')
Data = np.array(pickle.load(file))
Dim = [2, 5, 8]
Size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 450, 500]
Names = ['dim', 'size',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var',
         'RIS(The) est', 'MLE(The) est', 'MLE(Opt) est']


def obtain(ind, name):
    data = Data[Data[:, 0] == ind]
    x = data[:, 1]
    y = data[:, Names.index(name)]
    return x, y


def main():
    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs = axs.flatten()
    for i, dim in enumerate(Dim):
        x, y = obtain(dim, 'IS a-var')
        axs[2*i].loglog(x, np.sqrt(y / 20000), label='IS a-err')
        x, y = obtain(dim, 'NIS a-var')
        axs[2*i].loglog(x, np.sqrt(y / 20000), label='NIS a-err')
        x, y = obtain(dim, 'MIS a-var')
        axs[2*i].loglog(x, np.sqrt(y / 20000), label='MIS a-err')
        x, y = obtain(dim, 'RIS(O) a-var')
        axs[2*i].loglog(x, np.sqrt(y / 20000), label='RIS(O) a-err')
        x, y = obtain(dim, 'RIS(R) a-var')
        axs[2 * i].loglog(x, np.sqrt(y / 20000), label='RIS(R) a-err')
        x, y = obtain(dim, 'RIS(L) a-var')
        axs[2 * i].loglog(x, np.sqrt(y / 20000), label='RIS(L) a-err')

        x, y = obtain(dim, 'IS a-var')
        axs[2*i+1].loglog(x, np.sqrt(y / 20000), label='IS a-err')
        x, y = obtain(dim, 'NIS est')
        axs[2*i+1].loglog(x, np.abs(y - 1), label='NIS est')
        x, y = obtain(dim, 'MIS est')
        axs[2*i+1].loglog(x, np.abs(y - 1), label='MIS est')
        x, y = obtain(dim, 'RIS(O) est')
        axs[2*i+1].loglog(x, np.abs(y - 1), label='RIS(O) est')
        x, y = obtain(dim, 'RIS(R) est')
        axs[2 * i + 1].loglog(x, np.abs(y - 1), label='RIS(R) est')
        x, y = obtain(dim, 'RIS(L) est')
        axs[2 * i + 1].loglog(x, np.abs(y - 1), label='RIS(L) est')
        x, y = obtain(dim, 'MLE(Opt) est')
        axs[2*i+1].loglog(x, np.abs(y - 1), label='MLE(Opt) est')

    for ax in axs:
        ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
