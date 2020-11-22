import sys

import pandas as pd
from sklearn.impute import SimpleImputer

from prepare_train_test_sets import train_test_sets, load_housing_data


def features_targets():
    train = train_test_sets()[0]
    test = train_test_sets()[1]
    # part of `Prepare the data for Machine Learning algorithms`
    # in http://localhost:8888/notebooks/02_end_to_end_machine_learning_project.ipynb
    # separate the predictors and the labels
    f = train.drop("median_house_value", axis=1)
    t = train["median_house_value"].copy()
    return [f, t]


if __name__ == '__main__':
    housing = load_housing_data()
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    print("imputer stats {}".format(imputer.statistics_))
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                              index=housing_num.index)
    housing_cat = housing[["ocean_proximity"]]
    print(housing_cat.head(10))
    sys.exit(0)
