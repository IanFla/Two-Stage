import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('SizeBw7', 'rb')
Data = np.array(pickle.load(file))
Sizes = [50, 100, 150, 200, 500, 1000, 1500, 2000, 2500]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['size', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(size, name, to_ax, log=False):
    if size not in Sizes:
        print('size error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[Data[:, 0] == size]
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
        labels = ['size=' + str(size) for size in Sizes]
        if (i >= 2) and (i <= 4):
            axs[i].plot(Bw, np.log(Data[0, Names.index('IS a-var')]) * np.ones(Bw.size), c='k')
            labels = ['reference'] + labels

        for size in Sizes:
            if name == 'RIS(O) a-var/MIS a-var':
                data = Data[Data[:, 0] == size]
                x = data[:, 1]
                y = np.log(data[:, Names.index('RIS(O) a-var')] / data[:, Names.index('MIS a-var')])
                axs[i].plot(x, y)
            else:
                draw(size=size, name=name, to_ax=axs[i], log=True)

        axs[i].legend(labels)
        axs[i].set_title('log('+name+')')

    plt.show()


if __name__ == '__main__':
    draw_main()
