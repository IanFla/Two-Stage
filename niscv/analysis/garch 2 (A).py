import numpy as np
import pickle


def read(num):
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/garch/garch' + str(num), 'rb')
    data = pickle.load(file)
    data = np.array([[da[0] for da in dat] for dat in data])
    return data


def main():
    data = np.vstack([read(num) for num in np.arange(1, 3)])
    means = data.mean(axis=0)
    nvars = data.var(axis=0)
    nvars[:, 0] = 2000 * 3000 * nvars[:, 0]
    nvars[:, 1:] = 400000 * nvars[:, 1:]
    print(means[:, 0::2])
    print(nvars[:, 2::2] / nvars[:, 0].reshape([-1, 1]))
    print(means[:, 3::2] / means[:, 1].reshape([-1, 1]))
    return data


if __name__ == '__main__':
    R = main()
