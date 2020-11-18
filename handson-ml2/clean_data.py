import sys

from prepare_train_test_sets import train_test_sets

if __name__ == '__main__':
    train = train_test_sets()[0]
    test = train_test_sets()[1]
    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()
    sys.exit(0)
