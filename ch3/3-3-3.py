import numpy as np

X = np.array([1, 2])
print("X = ")
print(X)
print("X.shape = ", X.shape)

W = np.array([[1, 3, 5], [3, 4, 6]])
print("W  = ")
print(W)
print("W.shape = ", print(W.shape))

Y = np.dot(X, W)
print(Y)
