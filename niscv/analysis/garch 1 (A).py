import numpy as np
from matplotlib import pyplot as plt
import pickle


def read():
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/garch/garch_bw', 'rb')
    data = pickle.load(file)
    data = np.array([[da[0] for da in dat] for dat in data])
    return data


def main():
    data = read()
    avar0 = data[:, :, -5]
    avar1 = data[:, :, -3]
    avar2 = data[:, :, -1]
    BW = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    plt.semilogy(BW, avar0[:, 0], label='avar0')
    plt.semilogy(BW, avar0[:, 1], label='avar0')
    plt.semilogy(BW, avar1[:, 0], label='avar1')
    plt.semilogy(BW, avar1[:, 1], label='avar1')
    plt.semilogy(BW, avar2[:, 0], label='avar2')
    plt.semilogy(BW, avar2[:, 1], label='avar2')
    plt.legend()
    plt.show()
    plt.semilogy(BW, avar0[:, 2], label='avar0')
    plt.semilogy(BW, avar0[:, 3], label='avar0')
    plt.semilogy(BW, avar1[:, 2], label='avar1')
    plt.semilogy(BW, avar1[:, 3], label='avar1')
    plt.semilogy(BW, avar2[:, 2], label='avar2')
    plt.semilogy(BW, avar2[:, 3], label='avar2')
    plt.legend()
    plt.show()
    plt.semilogy(BW, avar0[:, 4], label='avar0')
    plt.semilogy(BW, avar0[:, 5], label='avar0')
    plt.semilogy(BW, avar1[:, 4], label='avar1')
    plt.semilogy(BW, avar1[:, 5], label='avar1')
    plt.semilogy(BW, avar2[:, 4], label='avar2')
    plt.semilogy(BW, avar2[:, 5], label='avar2')
    plt.legend()
    plt.show()
    # 1.3, 1.4, 1.5


if __name__ == '__main__':
    main()
