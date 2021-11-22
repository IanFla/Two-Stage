import numpy as np
import multiprocessing
import scipy.stats as st


def random(it):
    return st.norm.rvs()


def main():
    with multiprocessing.Pool(processes=10) as pool:
        its = np.arange(10)
        result = pool.map(random, its)

    print(result)


if __name__ == '__main__':
    main()
