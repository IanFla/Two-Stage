import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('Alpha0Bw7', 'rb')
Data = np.array(pickle.load(file))
Alpha0 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['alpha0', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(alpha0, name, to_ax, log=False):
    if alpha0 not in Alpha0:
        print('alpha0 error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[Data[:, 0] == alpha0]
    x = data[:, 1]
    y = data[:, Names.index(name)]
    if log:
        y = np.log(y)
        name = 'log(' + name + ')'

    to_ax.plot(x, y, label=name)
    return x, y


def draw_main():
    f, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs = axs.flatten()
    names = ['MIS a-var', 'RIS(O) a-var', 'RIS(O) a-var/MIS a-var']
    for i, name in enumerate(names):
        labels = ['alpha0=' + str(alpha0) for alpha0 in Alpha0]
        if i == 0:
            axs[i].plot(Bw, np.log(Data[0, Names.index('IS a-var')]) * np.ones(Bw.size), c='k')
            axs[i].plot(Bw, np.log(Data[Data[:, 0] == Alpha0[0], Names.index('NIS a-var')]), c='k')
            labels = ['reference 1', 'NIS a-var'] + labels

        for alpha0 in Alpha0:
            if name == 'RIS(O) a-var/MIS a-var':
                data = Data[Data[:, 0] == alpha0]
                x = data[:, 1]
                y = np.log(data[:, Names.index('RIS(O) a-var')] / data[:, Names.index('MIS a-var')])
                axs[i].plot(x, y)
            else:
                draw(alpha0=alpha0, name=name, to_ax=axs[i], log=True)

        axs[i].legend(labels)
        axs[i].set_title('log('+name+')')

    plt.show()


if __name__ == '__main__':
    draw_main()
