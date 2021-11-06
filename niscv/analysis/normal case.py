import numpy as np
from matplotlib import pyplot as plt
import pickle


def read(dim):
    file = open('niscv/data/normal_' + str(dim) + 'D', 'rb')
    data = np.array(pickle.load(file))
    data = data[0]
    print(type(data))
    data = np.array(data)
    print(data.shape)


def main():
    read(3)


if __name__ == '__main__':
    main()
