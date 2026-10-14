import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

class MetaLearnerTraditional:
    """
    Trains traditional regression models on meta-features to predict model performance.
    """
    def __init__(self, model_type='RandomForest', **kwargs):
        self.model_type = model_type
        if model_type == 'LinearRegression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown traditional model_type: {model_type}")
        
    def train(self, X_meta, y_perf):
        """
        Train the meta-learner.
        
        Args:
            X_meta (pd.DataFrame or np.ndarray): Meta-features.
            y_perf (pd.Series or np.ndarray): Performance metric (e.g., accuracy).
        """
        logger.info(f"Training traditional meta-learner ({self.model_type})...")
        self.model.fit(X_meta, y_perf)
        logger.info("Training complete.")

    def predict(self, X_meta):
        """
        Predict the performance for new datasets.
        
        Args:
            X_meta (pd.DataFrame or np.ndarray): Meta-features.
            
        Returns:
            np.ndarray: Predicted performance metrics.
        """
        logger.info(f"Predicting performance using {self.model_type}...")
        return self.model.predict(X_meta)
