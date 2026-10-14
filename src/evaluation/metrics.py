import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

def evaluate_classification(y_true, y_pred):
    """Evaluate classification predictions."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics

def evaluate_regression(y_true, y_pred):
    """Evaluate regression predictions."""
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    return metrics

def evaluate_clustering(X, labels):
    """Evaluate clustering using Silhouette Score."""
    try:
        score = silhouette_score(X, labels)
    except ValueError:
        score = -1 # Handle edge-cases like 1 cluster
    return {'silhouette_score': score}

def evaluate_model(y_true_or_X, y_pred_or_labels, task_type):
    """
    Unified function to evaluate model performance based on task type.
    
    Args:
        y_true_or_X: True labels (Classification/Regression) or Feature matrix (Clustering).
        y_pred_or_labels: Predicted labels (Classification/Regression/Clustering).
        task_type: One of 'classification', 'regression', 'clustering'.
    
    Returns:
        dict: Performance metrics.
    """
    logger.info(f"Evaluating model for {task_type} task...")
    
    if task_type == 'classification':
        return evaluate_classification(y_true_or_X, y_pred_or_labels)
    elif task_type == 'regression':
        return evaluate_regression(y_true_or_X, y_pred_or_labels)
    elif task_type == 'clustering':
        return evaluate_clustering(y_true_or_X, y_pred_or_labels)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
