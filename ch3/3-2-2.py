import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)
plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, linestyle="--", label="step")  # 波線で描画
plt.title('sigmoid and step')
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.legend()
plt.show()
