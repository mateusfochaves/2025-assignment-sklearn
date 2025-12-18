"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = validate_data(self, X, y, reset=True)
        check_classification_targets(y)

        if not isinstance(self.n_neighbors, (int, np.integer)):
            raise ValueError("Number of neighbors must be an integer.")
        if self.n_neighbors <= 0:
            raise ValueError("Number of neighbors must be positive")

        n_samples = X.shape[0]
        if self.n_neighbors > n_samples:
            raise ValueError(
                f"n_samples = {n_samples} should be >= "
                f"n_neighbors = {self.n_neighbors}."
            )

        self.X_fitted_ = X
        self.y_fitted_ = y

        if y.ndim == 1:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = [np.unique(y[:, j]) for j in range(y.shape[1])]

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """

        check_is_fitted(
            self, attributes=["X_fitted_", "y_fitted_", "classes_"]
        )
        X = validate_data(self, X, reset=False)

        dist = pairwise_distances(X, self.X_fitted_, metric="euclidean")
        k = int(self.n_neighbors)

        if k == 1:
            nearest = np.argmin(dist, axis=1)
            return self.y_fitted_[nearest]

        neigh_idx = np.argpartition(dist, kth=k-1, axis=1)[:, :k]

        y_train = self.y_fitted_
        n_test = X.shape[0]

        if y_train.ndim == 1:
            y_pred = np.empty(n_test, dtype=y_train.dtype)
            for i in range(n_test):
                votes = y_train[neigh_idx[i]]
                values, counts = np.unique(votes, return_counts=True)
                y_pred[i] = values[np.argmax(counts)]
            return y_pred

        n_outputs = y_train.shape[1]
        y_pred = np.empty((n_test, n_outputs), dtype=y_train.dtype)
        for i in range(n_test):
            votes = y_train[neigh_idx[i], :]
            for j in range(n_outputs):
                values, counts = np.unique(votes[:, j], return_counts=True)
                y_pred[i, j] = values[np.argmax(counts)]
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """

        check_is_fitted(
            self, attributes=["X_fitted_", "y_fitted_", "classes_"]
        )
        X, y = validate_data(self, X, y, reset=False)

        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """

        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be a pandas Datatime.")
            dt = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError("Time is not a column of X.")
            dt = X[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(dt):
                raise ValueError(
                    "Time column must be a datetime d type column."
                )

        months = dt.to_period("M") if isinstance(dt, pd.DatetimeIndex) else \
            dt.dt.to_period("M")
        ordered_months = pd.PeriodIndex(months.unique()).sort_values()

        return max(len(ordered_months)-1, 0)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """

        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be a pandas DatetimeIndex.")
            dt = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError("Time column is not a column of X.")
            dt = X[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(dt):
                raise ValueError(
                    "Time column must be a datetime dtype column."
                )

        months = dt.to_period("M") if isinstance(dt, pd.DatetimeIndex) else \
            dt.dt.to_period("M")
        ordered_months = pd.PeriodIndex(months.unique()).sort_values()

        n_samples = X.shape[0]
        n_splits = self.get_n_splits(X, y, groups)
        for i in range(n_splits):
            idx_train = range(n_samples)
            idx_test = range(n_samples)
            train_month = ordered_months[i]
            test_month = ordered_months[i+1]

            idx_train = np.flatnonzero(months == train_month)
            idx_test = np.flatnonzero(months == test_month)

            yield (idx_train, idx_test)
