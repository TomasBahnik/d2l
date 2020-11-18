import sys

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from prepare_train_test_sets import load_housing_data

if __name__ == '__main__':
    housing = load_housing_data()
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #              s=housing["population"] / 100, label="population", figsize=(10, 7),
    #              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    #              )
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.legend()
    plt.show()
    sys.exit(0)
