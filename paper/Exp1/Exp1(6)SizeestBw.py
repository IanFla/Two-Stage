import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('Data/SizeestBw7', 'rb')
Data = np.array(pickle.load(file))
Size_est = [1000, 2000, 5000, 10000, 15000, 30000, 60000, 100000, 150000, 200000]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['size', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(size_est, name, to_ax, log=False):
    if size_est not in Size_est:
        print('size error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[Data[:, 0] == size_est]
    x = data[:, 1]
    y = data[:, Names.index(name)]
    if log:
        y = np.log(y)
        name = 'log(' + name + ')'

    to_ax.plot(x, y, label=name)
    return x, y


def draw_main():
    f, axs = plt.subplots(3, 2, figsize=(20, 12))
    axs = axs.flatten()
    names = ['sqrt(ISE/Rf)', 'KLD', 'NIS a-var', 'MIS a-var', 'RIS(O) a-var', 'RIS(O,u) a-var']
    for i, name in enumerate(names):
        labels = ['size_est=' + str(size_est) for size_est in Size_est]
        if (i >= 2) and (i <= 4):
            axs[i].plot(Bw, np.log(Data[0, Names.index('IS a-var')]) * np.ones(Bw.size), c='k')
            labels = ['reference'] + labels

        for size_est in Size_est:
            draw(size_est=size_est, name=name, to_ax=axs[i], log=True)

        axs[i].legend(labels)
        axs[i].set_title('log('+name+')')

    plt.show()


if __name__ == '__main__':
    draw_main()
