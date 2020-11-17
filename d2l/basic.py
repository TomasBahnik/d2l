from mxnet import nd
from matplotlib import pyplot as plt
import random
import numpy as np
import math

# number of outcomes - sample space
# events are subsets of the sample space, elementary events are individual outcomes
# random variable is, in this case, identity function X(zero) = 0.., X(two) = 2 ...
STOP = 101  # 100001
n = 6  # outcomes are 0...n-1
probabilities = nd.ones(n) / n
number_of_tosses = 10000
figure_format = 'svg'


def multinomial():
    m = nd.random.multinomial(probabilities)
    print('single value :', m)
    print('10 values : ', nd.random.multinomial(probabilities, shape=number_of_tosses))
    print('5x10 values', nd.random.multinomial(probabilities, shape=(5, number_of_tosses)))


def rolling():
    tosses = nd.random.multinomial(probabilities, shape=number_of_tosses)
    # nth element contains history of X(n) = n
    toss_history = nd.zeros((n, number_of_tosses))
    # nth element contains total count of X(n) = n
    toss_cumulative_counts = nd.zeros(n)
    for i, toss in enumerate(tosses):
        toss_cumulative_counts[int(toss.asscalar())] += 1
        toss_history[:, i] = toss_cumulative_counts
    x = nd.arange(number_of_tosses).reshape((1, number_of_tosses)) + 1
    # each columns is divided by current number of tosses
    # gives current estimated probability
    estimates = toss_history / x
    plot_figure(estimates)
    print_estimated_pr(estimates)


def print_estimated_pr(estimates):
    probabilities_after_tosses(number_of_tosses // 1000, estimates)
    probabilities_after_tosses(number_of_tosses // 100, estimates)
    probabilities_after_tosses(number_of_tosses // 10, estimates)
    probabilities_after_tosses(number_of_tosses, estimates)


def uniform_sampling():
    counts = np.zeros(100)
    fig, axes = plt.subplots(2, 2, sharex=True)
    axes = axes.flatten()
    # Mangle subplots such that we can index them in a linear fashion rather than
    # a 2D grid
    for i in range(1, STOP):
        randint = random.randint(0, 99)
        counts[randint] += 1
        if i in [10, 101]:
            axes[int(math.log10(i)) - 2].bar(np.arange(1, 101), counts)
            plt.show()
        # plt.savefig('uniform_sampling.' + figure_format, dpi=300)


def plot_figure(estimated_pr):
    set_figure_size((12, 8))
    for i in range(6):
        plt.plot(estimated_pr[i, :].asnumpy(), label=("Pr(die=" + str(i) + ")"))
    plt.axhline(y=0.16666, color='black', linestyle='dashed')
    plt.legend()
    plt.savefig('die_probabilities.' + figure_format, dpi=300)
    plt.show()


def probabilities_after_tosses(tosses, estimated_pr):
    print('after {} tosses : {}'.format(tosses, estimated_pr[:, tosses - 1]))


def set_figure_size(figsize=(3.5, 2.5)):
    """Change the default figure size"""
    plt.rcParams['figure.figsize'] = figsize


def arrays():
    z = nd.zeros((1, 3))
    print('z zeroes', z)
    x = [.30, .60, .10]
    y = x[:1]
    z[:, 0] = x[:1]
    print('y', y)
    print('z', z)


def main():
    uniform_sampling()


if __name__ == '__main__':
    main()
