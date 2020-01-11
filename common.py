from mxnet import np


def model_errors(w, true_w, b, true_b):
    rw = w.reshape(true_w.shape)
    aew = true_w - rw
    rew = aew / rw * 100
    aeb = true_b - b
    reb = aeb / b * 100
    print('==== model errors ====')
    print('weights = {}, true weights = {}'.format(rw, true_w))
    print('bias = {}, true bias = {}'.format(b, true_b))
    print('Absolute error of weights : {}'.format(aew))
    print('Absolute error of bias : {}'.format(aeb))
    print('Relative error of weights : {} %'.format(rew))
    print('Relative error of bias : {} %'.format(reb))


def synthetic_data_local(w, b, num_examples):
    """generate y = X w + b + noise"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y
