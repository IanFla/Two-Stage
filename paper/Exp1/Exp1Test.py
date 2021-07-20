import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sb
import pandas as pd


Dim = [2, 4, 8]
Bw = np.around(np.linspace(0.6, 2.8, 12), 1)
A = [-1/4, -1/8, 0.0, 1/8, 1/4, 1/2, 1.0]
List = ['IS est', 'IS a-var', 'kernel number',
        'mean bdwth', 'ESS (kde)', 'sqrt(ISE/Rf)', 'KLD',
        'NIS est', 'NIS a-var', 'MIS est', 'MIS a-var',
        'CI mean', 'CI>30', 'R2(O)', 'R2(R)', 'R2(L)',
        'RIS(O) est', 'RIS(O) a-var', 'RIS(R) est', 'RIS(R) a-var', 'RIS(L) est', 'RIS(L) a-var',
        'RIS(O,u) est', 'RIS(O,u) a-var', 'RIS(R,u) est', 'RIS(R,u) a-var', 'RIS(L,u) est', 'RIS(L,u) a-var']


def draw(dim, name, log=False):
    select = List.index(name)
    data = Data[dim]
    matrix = np.zeros((len(A), len(Bw)))
    for i in range(len(A)):
        for j in range(len(Bw)):
            matrix[i, j] = data[i][j][select]

    if log:
        matrix = np.log(matrix)
        name = 'log(' + name + ')'

    df = pd.DataFrame(matrix, index=A, columns=Bw)
    f, ax = plt.subplots(figsize=(14, 7))
    sb.heatmap(df, annot=True, linewidths=0.5, ax=ax)
    ax.set_xlabel('h')
    ax.set_ylabel('a')
    maxi, maxj = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    mini, minj = np.unravel_index(np.argmin(matrix, axis=None), matrix.shape)
    ax.set_title(name + ': ({:.2g} [{},{}] <> {:.2g} [{},{}])'
                 .format(matrix.min(initial=1e10), A[mini], Bw[minj],
                         matrix.max(initial=-1e10), A[maxi], Bw[maxj]))
    plt.show()


if __name__ == '__main__':
    file = open('Ian', 'rb')
    Data = pickle.load(file)
    dimension = 8
    draw(Dim.index(dimension), 'mean bdwth', log=True)
    draw(Dim.index(dimension), 'CI>30')
    draw(Dim.index(dimension), 'sqrt(ISE/Rf)', log=True)  # 0.5, 1.2 / 0.25, 1.2 / 0.125, 1.4
    draw(Dim.index(dimension), 'KLD')  # 0.25, 0.8 / 0.125, 1.2 / 0, 1.2
    draw(Dim.index(dimension), 'NIS a-var', log=True)  # 0.25, 1 / 0.125, 1.2 / 0.125, 1.4
    draw(Dim.index(dimension), 'MIS a-var', log=True)  # 0.25, 0.8 / 0.125, 1.2 / 0.125, 1.4
    draw(Dim.index(dimension), 'RIS(O) a-var', log=True)  # 0, 2.8 / 0, 2.2 / 0, 1.4
