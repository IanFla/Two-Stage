import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('RatioBw7', 'rb')
Data = np.array(pickle.load(file))
Ratio = [10, 20, 30, 40, 60, 80, 100, 130, 160, 190]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['ratio', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(ratio, name, to_ax, log=False):
    if ratio not in Ratio:
        print('ratio error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[Data[:, 0] == ratio]
    x = data[:, 1]
    y = data[:, Names.index(name)]
    if log:
        y = np.log(y)
        name = 'log(' + name + ')'

    to_ax.plot(x, y, label=name)
    return x, y


def draw_main():
    f, axs = plt.subplots(3, 2, figsize=(16, 9))
    axs = axs.flatten()
    names = ['sqrt(ISE/Rf)', 'KLD', 'IS a-var', 'NIS a-var', 'MIS a-var', 'RIS(O) a-var']
    for i, name in enumerate(names):
        for ratio in Ratio:
            draw(ratio=ratio, name=name, to_ax=axs[i], log=True)

        axs[i].legend(['ratio=' + str(ratio) for ratio in Ratio])
        axs[i].set_title('log('+name+')')

    plt.show()


if __name__ == '__main__':
    draw_main()
