import numpy as np

x = np.array([
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
])
# Training data for NAND.
# y = np.array([1, 1, 1, 0])
# Training data for AND.
y = np.array([0, 0, 0, 1])
w = np.array([0.0, 0.0, 0.0])

eta = 1
for t in range(5):
    print("===== {} ======".format(t))
    for i in range(len(y)):
        y_pred = np.heaviside(np.dot(x[i], w), 0)
        error = (y[i] - y_pred)
        w = w + error * eta * x[i]
        # w += (y[i] - y_pred) * eta * x[i]
        print("{} : x[i]={}, error={}, w={}".format(i, x[i], error, w))

print("weights {}".format(w))
print("pred - train={}".format(np.heaviside(np.dot(x, w), 0) - y))
