

import os
import sys

os.chdir('/Users/edamame88/Projects/study/Deep01/deep-learning-from-scratch/ch03')

sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)


print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
