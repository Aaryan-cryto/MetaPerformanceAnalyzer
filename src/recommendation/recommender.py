import logging

logger = logging.getLogger(__name__)

class Recommender:
    """
    Recommends the best machine learning model based on predicted metrics.
    """
    def __init__(self, metric_to_optimize='accuracy'):
        """
        Args:
            metric_to_optimize (str): Metric used for ranking (e.g., 'accuracy' wants higher, 'rmse' wants lower).
        """
        self.metric_to_optimize = metric_to_optimize
        # Metrics that should be minimized
        self.minimize_metrics = ['rmse', 'mae', 'mse']

    def recommend(self, models_performance):
        """
        Takes predictions from meta-learner and recommends the best model.
        
        Args:
            models_performance (dict): Dictionary mapping model_name -> predicted metric score.
            
        Returns:
            tuple: (ranked_list_of_tuples, best_model_name)
        """
        logger.info(f"Generating recommendations optimizing for {self.metric_to_optimize}...")
        
        # Sort True if minimizing, False if maximizing
        reverse_sort = self.metric_to_optimize.lower() not in self.minimize_metrics
        
        ranked_models = sorted(
            models_performance.items(), 
            key=lambda x: x[1], 
            reverse=reverse_sort
        )
        
        if not ranked_models:
            logger.warning("No models provided for recommendation.")
            return [], None
            
        best_model = ranked_models[0][0]
        logger.info(f"Top recommendation: {best_model} with predicted {self.metric_to_optimize} {ranked_models[0][1]:.4f}")
        
        return ranked_models, best_model
