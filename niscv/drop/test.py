def experiment(seed, dim, target,
               init_proposal, size_est, x,
               size, ratio, resample,
               bw, factor, local, gamma, kdf, alpha0,
               alphaR, alphaL,
               stage=4, show=False):
    np.random.seed(seed)
    mle = MLE(dim, target, init_proposal, size_est=size_est, show=show)
    if stage >= 1:
        mle.disp('==IS==================================================IS==')
        mle.initial_estimation()
        if mle.show:
            mle.draw(mle.init_proposal, x=x, name='initial')

        mle.resampling(size=size, ratio=ratio, resample=resample)
        if stage >= 2:
            mle.disp('==NIS================================================NIS==')
            mle.proposal(bw=bw, factor=factor, local=local, gamma=gamma, kdf=kdf, alpha0=alpha0)
            Rf = target.pdf(target.rvs(size=size_est, random_state=seed)).mean()
            mle.nonparametric_estimation(Rf=Rf)
            if mle.show:
                mle.draw(mle.nonpar_proposal, x=x, name='nonparametric')

            if stage >= 3:
                mle.disp('==RIS================================================RIS==')
                mle.regression_estimation(alphaR=alphaR, alphaL=alphaL)
                if mle.show:
                    mle.draw(mle.mix_proposal, x=x, name='regression')

                if stage >= 4:
                    mle.disp('==MLE================================================MLE==')
                    mle.likelihood_estimation(opt=True, NR=True)

    return mle.result


def run(inputs):
    begin = dt.now()
    mean = np.zeros(inputs[0])
    target = mvnorm(mean=mean)
    init_proposal = mvnorm(mean=mean, cov=4)
    x = np.linspace(-4, 4, 101)
    print(inputs)
    result = experiment(seed=3033079628, dim=mean.size, target=target,
                        init_proposal=init_proposal, size_est=100000, x=x,
                        size=inputs[1], ratio=100, resample=True,
                        bw=1.0, factor='scott', local=False, gamma=1.0, kdf=0, alpha0=0.1,
                        alphaR=10000.0, alphaL=0.1,
                        stage=4, show=False)
    end = dt.now()
    print('Total spent: {}s (dim {} size {})'
          .format((end - begin).seconds, inputs[0], inputs[1]))
    return inputs + result


def main():
    Dim = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    Size = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 450, 500]
    inputs = []
    for dim in Dim:
        for size in Size:
            inputs.append([dim, size])

    pool = multiprocessing.Pool(2)
    results = pool.map(run, inputs)

    with open('DimSize', 'wb') as file:
        pickle.dump(results, file)
        file.close()