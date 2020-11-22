import os
import urllib.request

import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from pandas.plotting import scatter_matrix

from common import PROJECT_ROOT_DIR, IMAGES_PATH
from prepare_train_test_sets import split_train_test


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def calif_image(data):
    images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
    os.makedirs(images_path, exist_ok=True)
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    filename = "california.png"
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
    urllib.request.urlretrieve(url, os.path.join(images_path, filename))
    california_img = mpimg.imread(os.path.join(images_path, filename))
    ax = data.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                   s=data['population'] / 100, label="Population",
                   c="median_house_value", cmap=plt.get_cmap("jet"),
                   colorbar=False, alpha=0.4,
                   )
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    prices = data["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values / prices.max())
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)
    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()


def scatter_plots(data):
    data.plot(kind="scatter", x="longitude", y="latitude")
    save_fig("bad_visualization_plot")
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    save_fig("better_visualization_plot")
    # The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed).
    # This is a temporary fix (see: https://github.com/pandas-dev/pandas/issues/10611 ).
    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=data["population"] / 100, label="population", figsize=(10, 7),
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
              sharex=False)
    plt.legend()
    save_fig("housing_prices_scatterplot")


def looking_for_correlation(data):
    # global attributes
    corr_matrix = data.corr()
    print("correlation matrix\n{}".format(corr_matrix))
    print("correlation matrix - median_house_value \n{}".format(
        corr_matrix["median_house_value"].sort_values(ascending=False)))
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    scatter_matrix(data[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")
    data.plot(kind="scatter", x="median_income", y="median_house_value",
              alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    save_fig("income_vs_house_value_scatterplot")
    # Experimenting with Attribute Combinations
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    corr_matrix = data.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    data.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
              alpha=0.2)
    plt.axis([0, 5, 0, 520000])
    plt.show()
    data.describe()


def basic_info(data):
    data.head()
    data.info()
    data["ocean_proximity"].value_counts()
    data.describe()
    data.hist(bins=50, figsize=(20, 15))
    save_fig("attribute_histogram_plots")
    plt.show()
    # to make this notebook's output identical at every run
    np.random.seed(42)
    train_set, test_set = split_train_test(data, 0.2)
    len(train_set)
    len(test_set)
