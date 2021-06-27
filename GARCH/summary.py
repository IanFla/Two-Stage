import pickle
import numpy as np

file = open('Ian', 'rb')
data = np.array(pickle.load(file))
print(np.round(data.mean(axis=0), 5))
var = data.var(axis=0)
nvar = 100000 * var
nvar[0:2, 1] *= 15
nvar[2:4, 1] *= 20
nvar[4:6, 1] *= 30
for tmp in nvar:
    print(np.round(tmp, 5))

print(nvar[:, 1] / nvar[:, -1])
