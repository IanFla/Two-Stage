import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('Data/KdfBw7', 'rb')
Data = np.array(pickle.load(file))
Kdf = [3, 4, 5, 8, 12, 20, 50, 100, 0]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['kdf', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(kdf, name, to_ax, log=False):
    if kdf not in Kdf:
        print('kdf error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[Data[:, 0] == kdf]
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
    names = ['sqrt(ISE/Rf)', 'KLD', 'NIS a-var', 'MIS a-var', 'RIS(O) a-var', 'RIS(O) a-var/MIS a-var']
    for i, name in enumerate(names):
        labels = ['kdf=' + str(kdf) for kdf in Kdf]
        if (i >= 2) and (i <= 4):
            axs[i].plot(Bw, np.log(Data[0, Names.index('IS a-var')]) * np.ones(Bw.size), c='k')
            labels = ['reference'] + labels

        for kdf in Kdf:
            if name == 'RIS(O) a-var/MIS a-var':
                data = Data[Data[:, 0] == kdf]
                x = data[:, 1]
                y = np.log(data[:, Names.index('RIS(O) a-var')] / data[:, Names.index('MIS a-var')])
                axs[i].plot(x, y)
            else:
                draw(kdf=kdf, name=name, to_ax=axs[i], log=True)

        axs[i].legend(labels)
        axs[i].set_title('log('+name+')')

    plt.show()


if __name__ == '__main__':
    draw_main()
