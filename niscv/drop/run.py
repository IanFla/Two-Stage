import multiprocessing


def run(it):
    return it


if __name__ == '__main__':
    pool = multiprocessing.Pool(2)
    abb = [0, 1]
    R = pool.map(run, abb)
    print(R)
