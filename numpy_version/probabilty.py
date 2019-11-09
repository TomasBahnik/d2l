import sys
import d2l
from mxnet import np, npx, nd
import random

npx.set_np()

# theory
# number of outcomes - sample space
# events are subsets of the sample space, elementary events are individual outcomes
# random variable is, in the case of dice (also die), identity function X(zero) = 0.., X(two) = 2 ...

n = 6  # outcomes are 0...n-1

# this is an ASSUMPTION that we can't *mathematically* prove
# the correctness can be verified only by experiment like in physics
#fair_probs = [1.0 / n] * n  # <class 'list'>
fair_probs = np.ones(n) / n  # <class 'mxnet.numpy.ndarray'>
fair_probs_nd = nd.ones(n) / n # <class 'mxnet.ndarray.ndarray.NDArray'>
tosses = 100
experiments = 3

def multinomial_nd():
    m = nd.random.multinomial(fair_probs_nd)
    print('single value :', m)
    m = nd.random.multinomial(fair_probs_nd, shape=tosses)
    print("{} values {}".format(tosses, m))
    m = nd.random.multinomial(fair_probs_nd, shape=(experiments, tosses))
    print("{}x{} values {}\n{}".format(experiments, tosses, m, type(m)))


def multinomial_np():
    m = np.random.multinomial(tosses, fair_probs, size=experiments)
    print(m, m.shape, type(m))


def main():
    multinomial_np()
    # print("\n\n **** ND")
    # multinomial_nd()


if __name__ == '__main__':
    main()
    sys.exit(0)
