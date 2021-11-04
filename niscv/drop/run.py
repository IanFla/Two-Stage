import multiprocessing


if __name__ == '__main__':
    run = lambda it: it
    pool = multiprocessing.Pool(2)
    abb = [0, 1]
    R = pool.map(run, abb)
    print(R)
