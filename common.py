def model_errors(w, true_w, b, true_b):
    rw = w.reshape(true_w.shape)
    aew = true_w - rw
    rew = aew / rw * 100
    aeb = true_b - b
    reb = aeb / b * 100
    print('estimated w = {}, true w = {}'.format(rw, true_w))
    print('estimated b = {}, true b = {}'.format(b, true_b))
    print('Abs error in estimating w', aew)
    print('Abs error in estimating b', aeb)
    print('Relative error of w : {} %'.format(rew))
    print('Relative error of b : {} %'.format(reb))
