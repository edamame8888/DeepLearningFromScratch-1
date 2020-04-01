import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print("1層目の処理 ==================")

print("W1 = ")
print(W1)
print("W1.shape = ", W1.shape)

print("-----------------------------")

print("X = ")
print(X)
print("X.shape = ", X.shape)

print("-----------------------------")

print("B1 = ")
print(B1)
print("B1.shape = ", B1.shape)

print("-----------------------------")
A1 = np.dot(X, W1) + B1
print("A1 = ")
print(A1)
print("A1.shape = ", A1.shape)

print("-----------------------------")
Z1 = sigmoid(A1)
print("Z1 = ")
print(Z1)
print("Z1.shape = ", Z1.shape)
print("2層目の処理 ==================")

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])


print("-----------------------------")
print("W2 = ")
print(W2)
print("W2.shape = ", W2.shape)


print("-----------------------------")
print("B2 = ")
print(B2)
print("B2.shape = ", B2.shape)

print("-----------------------------")
A2 = np.dot(Z1, W2) + B2
print("A2 = ")
print(A2)
print("A2.shape = ", A2.shape)

print("-----------------------------")

Z2 = sigmoid(A2)
print("Z2 = ")
print(Z2)
print("Z2.shape = ", Z2.shape)

print("3層目の処理 ==================")


def identity_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])


print("W3 = ")
print(W3)
print("W3.shape = ", W3.shape)


print("-----------------------------")
print("B3 = ")
print(B3)
print("B3.shape = ", B3.shape)

print("-----------------------------")
A3 = np.dot(Z2, W3) + B3
print("A3 = ")
print(A3)
print("A3.shape = ", A3.shape)

print("-----------------------------")

Z3 = identity_function(A3)
print("Z3 = ")
print(Z3)
print("Z3.shape = ", Z3.shape)
