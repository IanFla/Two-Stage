import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('Data/DimGammaBw1', 'rb')
Data = np.array(pickle.load(file))
Dim = [2, 4, 6, 8, 10]
Gamma = [0.1, 0.3, 0.5, 1.0]
Bw = np.linspace(0.4, 3.2, 15)
Names = ['dim', 'gamma', 'bw',
         'IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(dim, gamma, name, to_ax, log=False):
    if dim not in Dim:
        print('dim error')
        return

    if gamma not in Gamma:
        print('gamma error')
        return

    if name not in Names:
        print('name error')
        return

    data = Data[(Data[:, 0] == dim) & (Data[:, 1] == gamma)]
    x = data[:, 2]
    y = data[:, Names.index(name)]
    if log:
        y = np.log(y)
        name = 'log(' + name + ')'

    to_ax.plot(x, y, label=name)
    return x, y


def draw_main(dim):
    f, axs = plt.subplots(3, 2, figsize=(20, 12))
    axs = axs.flatten()
    for gamma in Gamma:
        draw(dim=dim, gamma=gamma, name='sqrt(ISE/Rf)', to_ax=axs[0], log=True)
        draw(dim=dim, gamma=gamma, name='KLD', to_ax=axs[1], log=True)

    labels = ['gamma='+str(gamma) for gamma in Gamma]
    axs[0].legend(labels)
    axs[0].set_title('log(sqrt(ISE/Rf))')
    axs[1].legend(labels)
    axs[1].set_title('log(KLD)')
    for i, gamma in enumerate(Gamma):
        draw(dim=dim, gamma=gamma, name='IS a-var', to_ax=axs[i+2], log=True)
        draw(dim=dim, gamma=gamma, name='NIS a-var', to_ax=axs[i+2], log=True)
        draw(dim=dim, gamma=gamma, name='MIS a-var', to_ax=axs[i+2], log=True)
        draw(dim=dim, gamma=gamma, name='RIS(O) a-var', to_ax=axs[i+2], log=True)
        draw(dim=dim, gamma=gamma, name='RIS(O,u) a-var', to_ax=axs[i + 2], log=True)
        axs[i+2].legend()
        axs[i+2].set_title('gamma='+str(gamma))

    plt.show()


if __name__ == '__main__':
    for d in Dim:
        draw_main(dim=d)
