from datetime import datetime as dt
import multiprocessing
import os

def run(it):
    print(it)
    return est(100000000)

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    with multiprocessing.Pool(processes=32) as pool:
        begin = dt.now()
        its = np.arange(1000)
        R = pool.map(run, its)
        end = dt.now()
        print((end - begin).seconds)

    with open('../data/garch/p_acc', 'wb') as file:
        pickle.dump(R, file)

def main():
    file = open('/Users/ianfla/Documents/GitHub/Two-Stage/niscv/data/garch/p_acc', 'rb')
    data = pickle.load(file)
    data = np.array([da[0] for da in data])
    print(data.mean(), data.std())