import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

def train_linear_regression(X_train, y_train, **kwargs):
    """Train a Linear Regression model."""
    logger.info("Training Linear Regression model...")
    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, **kwargs):
    """Train a Decision Tree Regressor."""
    logger.info("Training Decision Tree Regressor...")
    model = DecisionTreeRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model

def get_regression_models():
    """Return a dictionary of uninitialized regression models."""
    return {
        'LinearRegression': LinearRegression,
        'DecisionTree': DecisionTreeRegressor
    }
