# Discover and visualize the data to gain insights
import sys

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from prepare_train_test_sets import load_housing_data


def corr(data):
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    corr_matrix = data.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))


def scatter(data):
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(data[attributes], figsize=(12, 8))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    housing = load_housing_data()
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #              s=housing["population"] / 100, label="population", figsize=(10, 7),
    #              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    #              )
    # scatter(housing)
    corr(housing)
    sys.exit(0)
