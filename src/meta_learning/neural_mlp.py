import logging
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class MetaLearnerMLP:
    """
    Neural Meta-Learning model using Multi-Layer Perceptron (MLP) Regressor.
    Includes proper scaling of meta-features.
    """
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, **kwargs):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.scaler = StandardScaler()

    def train(self, X_meta, y_perf):
        """
        Train the MLP meta-learner.
        
        Args:
            X_meta (pd.DataFrame or np.ndarray): Meta-features.
            y_perf (pd.Series or np.ndarray): Performance metrics.
        """
        logger.info("Scaling meta-features for MLP...")
        X_scaled = self.scaler.fit_transform(X_meta)
        
        logger.info("Training MLP meta-learner...")
        self.model.fit(X_scaled, y_perf)
        logger.info("Training complete.")

    def predict(self, X_meta):
        """
        Predict performance using MLP meta-learner.
        
        Args:
            X_meta (pd.DataFrame or np.ndarray): Meta-features for prediction.
            
        Returns:
            np.ndarray: Predicted performance.
        """
        logger.info("Scaling input meta-features...")
        X_scaled = self.scaler.transform(X_meta)
        
        logger.info("Predicting performance using MLP...")
        return self.model.predict(X_scaled)
