"""
A script (train.py) to train the model(s).
The script accepts arguments for input (dataset) and output folders (model pickles)
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from mle_lib.api import (
    DEFAULT_ARTIFACTS_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_PATH,
    logger,
    read,
    write,
)


def income_cat_proportions(data):
    """
    Computes the proportion of all income categories in the data.

    Parameters:
    -----------
    data:
        pd.DataFrame, Data with income_categories
    """
    return data["income_cat"].value_counts() / len(data)


def load_data(input_path=DEFAULT_DATA_PATH):
    """
    Loads the data from given path

    Parameters:
    -----------
    input_path:
        Path where data is stored
    """
    # declaring global variables
    global housing_raw, train_set, test_set, strat_train_set, strat_test_set
    raw_path = os.path.join(input_path, "raw", "housing.csv")
    train_set_path = os.path.join(input_path, "train", "housing_train.csv")
    test_set_path = os.path.join(input_path, "test", "housing_test.csv")
    strat_train_set_path = os.path.join(
        input_path, "train", "housing_train_income_stratified.csv"
    )
    strat_test_set_path = os.path.join(
        input_path, "test", "housing_test_income_stratified.csv"
    )

    # reading data
    logger.info("Reading Data files")
    housing_raw = read(raw_path)
    train_set = read(train_set_path)
    test_set = read(test_set_path)
    strat_train_set = read(strat_train_set_path)
    strat_test_set = read(strat_test_set_path)


def process_data(input_path=DEFAULT_DATA_PATH, return_train=False):
    """
    Process and ready the data for training and testing

    Parameters:
    -----------
    input_path:
        Path where data is stored

    Returns:
    --------
    Features and target data of train and test sets

    """
    global housing_labels, housing_prepared
    load_data(input_path)
    logger.info("Data Loaded")
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing_raw),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    logger.info("Processing train data")

    housing.plot(kind="scatter", x="longitude", y="latitude", backend="matplotlib")
    plt.show()
    plt.savefig(os.path.join(DEFAULT_RESULTS_PATH, "long_vs_lat.png"))
    housing.plot(
        kind="scatter", x="longitude", y="latitude", alpha=0.1, backend="matplotlib"
    )
    plt.show()
    plt.savefig(os.path.join(DEFAULT_RESULTS_PATH, "long_vs_lat_transparent.png"))

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    logger.info("Train data Imputing")
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)
    # imputing missing values
    imputer.fit(housing_num)

    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    # getting dummy variables
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    logger.info("Processing test data")
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    logger.info("Test Data Imputing")
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    if not return_train:
        return X_test_prepared, y_test
    else:
        return housing_prepared, housing_labels, X_test_prepared, y_test


def train_models(output_path=DEFAULT_ARTIFACTS_PATH):
    """
    Trains models and saves the objects

    Parameters:
    ----------
    output_path:
        Path to save the model objects
    """
    # training linear regression
    logger.info("Training Linear Regression")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    filename_lin = os.path.join(output_path, "Linear_Regression.pkl")
    write(lin_reg, filename_lin)

    # training decision tree
    logger.info("Training Decision Tree Regressor")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    filename_tree = os.path.join(output_path, "Decision_Tree_Regressor.pkl")
    write(tree_reg, filename_tree)

    # random search CV
    logger.info("Training Random forest and tuning parameters using random search")
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    filename_random = os.path.join(
        output_path, "random_search_best_estimator_forest.pkl"
    )
    write(rnd_search.best_estimator_, filename_random)

    feature_importances = rnd_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    write(
        pd.DataFrame(feature_importances),
        os.path.join(DEFAULT_RESULTS_PATH, "random_search_feature_importances.csv"),
    )

    logger.info("Training Random forest and tuning parameters using grid search")
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    filename_grid = os.path.join(output_path, "grid_search_best_estimator_forest.pkl")
    write(grid_search.best_estimator_, filename_grid)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    write(
        pd.DataFrame(feature_importances),
        os.path.join(DEFAULT_RESULTS_PATH, "grid_search_feature_importances.csv"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Data Path")
    parser.add_argument("--output_path", help="Path to save model objects")
    args = parser.parse_args()

    if args.input_path is not None:
        X_test, y_test = process_data(args.input_path)
    else:
        X_test, y_test = process_data()

    if args.output_path is not None:
        train_models(args.output_path)
    else:
        train_models()

