import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# Order of Class is used as label of class
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from prepare_data_ml_alg import features_labels
from prepare_train_test_sets import load_housing_data, fetch_housing_data
from prepare_train_test_sets import split_data
from prepare_train_test_sets import strata

fetch_housing_data()
housing = load_housing_data()

housing = strata(housing)
strat_train_set, strat_test_set = split_data(housing)

# Now you should remove the income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Prepare the data for Machine Learning algorithms
housing, housing_labels = features_labels(strat_train_set)
housing_num = housing.drop("ocean_proximity", axis=1)

# Handling Text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]

# Let's create a custom transformer to add extra attributes:
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Now let's build a pipeline for pre-processing the numerical attributes:
# As with all the transformations, it is important to fit the scalers to the training data only,
# not to the full dataset (including the test set).
# Only then can you use them to transform the training set and the test set (and new data).

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# # Select and train a model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full pre-processing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

# Compare against the actual values:
print("Labels:", list(some_labels))

# My
# Predictions: [210644.60459286 317768.80697211 210956.43331178  59218.98886849  189747.55849879]
# Labels:      [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]

# Book
# Predictions: [210644.6045     317768.8069     210956.4333      59218.9888      189747.5584]
# Labels:      [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear model RMS error={}".format(lin_rmse))

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("Linear model mean absolute error={}".format(lin_mae))

tree_reg = DecisionTreeRegressor()
t0 = time.time()
tree_reg.fit(housing_prepared, housing_labels)
t1 = time.time()
print("Decision Tree model RMS time={}".format(t1 - t0))

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree model RMS error={}".format(tree_rmse))


# Scikit-Learn’s cross-validation features expect a utility function (greater is better)
# rather than a cost function (lower is better),
def cross_validation(model, features, labels):
    print("Model type : {}".format(type(model)))
    start = time.time()
    scores = cross_val_score(model, features, labels, scoring="neg_mean_squared_error",
                             cv=10)
    end = time.time()
    print("Model cross validation score took {} sec.".format(end - start))
    rmse_scores = np.sqrt(-scores)
    print("Model RMS error scores")
    display_scores(rmse_scores)
    print("Model RMS error scores stats : {}".format(pd.Series(rmse_scores).describe()))


# pandas series std differs from ndarray std
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


cross_validation(lin_reg, housing_prepared, housing_labels)

cross_validation(tree_reg, housing_prepared, housing_labels)

# **Note**: we specify `n_estimators=100` to be future-proof since the default value is going to change to 100
# in Scikit-Learn 0.22 (for simplicity, this is not shown in the book).
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest model RMS error={}".format(forest_rmse))

# cross_validation(forest_reg, housing_prepared, housing_labels)

if __name__ == '__main__':
    sys.exit(0)
