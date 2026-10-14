import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def train_logistic_regression(X_train, y_train, **kwargs):
    """Train a Logistic Regression model."""
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, **kwargs):
    """Train a Support Vector Machine classifier."""
    logger.info("Training SVM model...")
    model = SVC(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    """Train a Random Forest classifier."""
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def get_classification_models():
    """Return a dictionary of uninitialized classification models."""
    return {
        'LogisticRegression': LogisticRegression,
        'SVM': SVC,
        'RandomForest': RandomForestClassifier
    }
