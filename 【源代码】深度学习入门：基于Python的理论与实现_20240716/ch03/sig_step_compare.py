# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def step_function(x):
    return np.array(x > 0, dtype=np.int64)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1, label='sigmoid')
plt.plot(x, y2, 'k--', label='step')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.title('Sigmoid vs Step Function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
