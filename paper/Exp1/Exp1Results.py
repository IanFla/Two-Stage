# unbalanced
# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=False,
#            bw=1.4, local=False, gamma=0.1, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1)
# 0.9876 - 2.2418 - 2.4158 - 1.9280

# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=False,
#            bw=1.5, local=False, gamma=0.1, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
# 1.0582 - 2.4180 - 2.6467

# resampling
# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.4, local=False, gamma=0.1, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1)
# 0.8044 - 1.0424 - 1.2463 - 0.0613

# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.2, local=False, gamma=0.1, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
# 0.6895 - 0.7563 - 0.9133

# locally adaptive bandwidth
# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.8, local=True, gamma=0.3, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1)
# 0.9053 - 1.8401 - 2.1353 - 0.1192

# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.4, local=True, gamma=0.3, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
# 0.7041 - 1.0171 - 1.1774

# adaptive bandwidth 1/d
# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=2.0, local=False, gamma=0.1, a=1/8, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1)
# 1.1491 - 1.6091 - 1.8619 - 0.1010

# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.3, local=False, gamma=0.1, a=1/8, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
# 0.7469 - 0.7186 - 0.8744

# adaptive bandwidth 1/2
# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=5.0, local=False, gamma=0.1, a=1/2, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1)
# 2.8728 - 3.4285 - 3.7686 - 0.4151

# experiment(seed=1234, dim=8, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=2.7, local=False, gamma=0.1, a=1/2, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
# 1.5513 - 1.9465 - 2.1714

# modified: *************
# dim = 8
# mean = np.zeros(dim)
# target = mvnorm(mean=mean)
# init_proposal = mvnorm(mean=mean, cov=4)
# experiment(seed=1234, dim=dim, target=target, init_proposal=init_proposal, size_est=100000,
#            size=1000, ratio=100, resample=True,
#            bw=1.4, local=True, gamma=0.3, a=0.0, rate=0.9,
#            alphaR=1000000.0, alphaL=0.1, stage=2)
