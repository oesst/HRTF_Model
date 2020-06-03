import numpy as np


a = np.ones((20,25,100))
b = np.zeros((20,25,100))

n_inds = 18
inds = np.random.randint(0,25*20,n_inds)

inds = np.unravel_index(inds,(20,25))
print(inds)

b[inds[0],inds[1],:] = a[inds[0],inds[1],:]
print(b.shape)

print(b)
