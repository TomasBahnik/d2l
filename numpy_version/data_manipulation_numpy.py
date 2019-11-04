from mxnet import np, npx


def main():
    npx.set_np()
    x = np.arange(12)
    print(x)


if __name__ == '__main__':
    main()
