import numpy as np

x = np.array([
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
])
# Training data for NAND.
y = np.array([1, 1, 1, 0])
# Training data for AND.
# y = np.array([0, 0, 0, 1])
w = np.array([0.0, 0.0, 0.0])

eta = 0.34
for t in range(100):
    y_pred = np.heaviside(np.dot(x, w), 0)
    error = (y - y_pred)
    w = w + np.dot(error * eta, x)

print("weights {}".format(w))
print("pred - train={}".format(np.heaviside(np.dot(x, w), 0) - y))
