import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 重みとバイアスだけが AND と違う!
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print("AND GATE")
print("AND(0, 0) =", AND(0, 0))
print("AND(1, 0) =", AND(1, 0))
print("AND(0, 1) =", AND(0, 1))
print("AND(1, 1) =", AND(1, 1))

print("NAND GATE")
print("NAND(0, 0) =", NAND(0, 0))
print("NAND(1, 0) =", NAND(1, 0))
print("NAND(0, 1) =", NAND(0, 1))
print("NAND(1, 1) =", NAND(1, 1))
print("OR GATE")
print("OR(0, 0) =", OR(0, 0))
print("OR(1, 0) =", OR(1, 0))
print("OR(0, 1) =", OR(0, 1))
print("OR(1, 1) =", OR(1, 1))
