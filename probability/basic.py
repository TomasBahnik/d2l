from IPython import display
import numpy as np
from mxnet import nd
import math
from matplotlib import pyplot as plt
import random

# number of outcomes - sample space
# events are subsets of the sample space, elementary events are individual outcomes
# random variable is, in this case, identity function X(zero) = 0.., X(two) = 2 ...
n = 6  # outcomes are 0...n-1
probabilities = nd.ones(n) / n
number_of_tosses = 100


def multinomial():
    m = nd.random.multinomial(probabilities)
    print('single value :', m)
    print('10 values : ', nd.random.multinomial(probabilities, shape=number_of_tosses))
    print('5x10 values', nd.random.multinomial(probabilities, shape=(5, number_of_tosses)))


def rolling():
    tosses = nd.random.multinomial(probabilities, shape=number_of_tosses)
    # nth element contains history of X(n) = n
    toss_history = nd.zeros((n, number_of_tosses))
    # nth element contains total count of of X(n) = n
    toss_cumulative_counts = nd.zeros(n)
    for i, toss in enumerate(tosses):
        toss_cumulative_counts[int(toss.asscalar())] += 1
        toss_history[:, i] = toss_cumulative_counts
    x = nd.arange(number_of_tosses).reshape((1, number_of_tosses)) + 1
    # each columns is divided by current number of tosses
    # gives current estimated probability
    estimates = toss_history / x
    probabilities_after_tosses(1, estimates)
    probabilities_after_tosses(number_of_tosses // 100, estimates)
    probabilities_after_tosses(number_of_tosses // 10, estimates)
    probabilities_after_tosses(number_of_tosses // 3, estimates)
    probabilities_after_tosses(number_of_tosses // 2, estimates)
    probabilities_after_tosses(number_of_tosses, estimates)
    set_figsize((6, 4))
    for i in range(6):
        plt.plot(estimates[i, :].asnumpy(), label=("Pr(die=" + str(i) + ")"))
    plt.axhline(y=0.16666, color='black', linestyle='dashed')
    plt.legend()
    plt.savefig('die_probabilities.png', dpi=300)
    plt.show()


def probabilities_after_tosses(tosses, estimated_pr):
    print('after {} tosses : {}'.format(tosses, estimated_pr[:, tosses - 1]))


# Save to the d2l package.
def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    display.set_matplotlib_formats('png')


# Save to the d2l package.
def set_figsize(figsize=(3.5, 2.5)):
    """Change the default figure size"""
    use_svg_display()
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
    rolling()
    # arrays()


if __name__ == '__main__':
    main()
