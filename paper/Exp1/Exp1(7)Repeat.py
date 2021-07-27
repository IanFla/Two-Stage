import numpy as np
from matplotlib import pyplot as plt
import pickle


file = open('Data/Repeat7', 'rb')
Data = np.array(pickle.load(file))
Repeat = np.arange(200).reshape([-1, 1]) + 1
Data -= Repeat
Bw = np.linspace(0.4, 3.2, 15)
Names = ['IS est', 'IS a-var', 'n0/ESS', 'n0/RSS', 'kernel number',
         'mean bdwth', 'kde ESS', 'sqrt(ISE/Rf)', 'KLD',
         'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
         'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
         'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
         'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def compare(name):
    ind = Names.index(name + ' est')
    Ests = Data[:, ind]
    Vars = Data[:, ind + 1]
    nMSE = 100000 * np.mean((Ests - 1) ** 2)
    aVar = 100000 * np.var(Ests)
    MHaVar = Vars.mean()
    print('{}: (nMSE: {}; aVar: {}; MeanHat(aVar): {}; nMSE/MeanHat(aVar): {})'
          .format(name, nMSE, aVar, MHaVar, nMSE / MHaVar))


def main():
    Select = ['IS', 'NIS', 'MIS', 'RIS(O)', 'RIS(R)', 'RIS(L)',
              'RIS(O,u)', 'RIS(R,u)', 'RIS(L,u)']
    for name in Select:
        compare(name)


if __name__ == '__main__':
    main()
