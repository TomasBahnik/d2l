import os
import sys
import tarfile
import urllib.request

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from common import PROJECT_ROOT_DIR

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def strata(data):
    data["income_cat"] = pd.cut(data["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
    return data


def split_data(data):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strata_test_set = None
    strata_train_set = None
    for train_index, test_index in split.split(data, data["income_cat"]):
        strata_train_set = data.loc[train_index]
        strata_test_set = data.loc[test_index]
    return [strata_train_set, strata_test_set]


def train_test_sets():
    fetch_housing_data()
    data = load_housing_data()
    # train_set, test_set = split_train_test(housing, 0.2)
    # housing_with_id = housing.reset_index()
    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    strata(data)
    # housing["income_cat"].hist()
    # plt.show()
    train_test = split_data(data)
    for set_ in train_test:
        set_.drop("income_cat", axis=1, inplace=True)
    return train_test


if __name__ == '__main__':
    train_test_sets()
    sys.exit(0)
