import sys


def features_labels(train_set):
    # revert to a clean training set and separate the predictors (features) and the labels
    features = train_set.drop("median_house_value", axis=1)  # drop labels for training set
    labels = train_set["median_house_value"].copy()
    return [features, labels]


if __name__ == '__main__':
    sys.exit(0)
