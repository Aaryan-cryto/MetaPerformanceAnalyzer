import logging
import numpy as np

logger = logging.getLogger(__name__)

class DifficultyScorer:
    """
    Computes a dataset difficulty score based on extracted meta-features.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_score(self, meta_features):
        """
        Computes the difficulty score.
        
        Args:
            meta_features (dict): Dictionary from MetaFeatureExtractor.
            
        Returns:
            tuple: (numerical_score, category)
        """
        self.logger.info("Computing dataset difficulty score...")
        
        # Base assumptions for heuristics:
        # - High feature-to-sample ratio -> harder
        # - High class imbalance -> harder
        # - High correlation -> can go either way, assume independent features are easier (lower correlation = easier)
        # - High variance -> generally easier to separate than low variance if it's signal, but here we'll just normalize its impact
        
        fsm_ratio = meta_features.get('feature_to_sample_ratio', 0)
        imbalance = meta_features.get('class_imbalance_ratio', 1)
        correlation = meta_features.get('mean_correlation', 0)
        
        # Heuristic scoring (normalized rough sums):
        # 1. Cap fsm_ratio at 1.0 (if features > samples, it's max difficulty from dimensionality)
        norm_fsm = min(fsm_ratio, 1.0)
        
        # 2. Imbalance: 1 is perfectly balanced. 10+ is highly imbalanced.
        # Normalize roughly with a logistic/cap curve
        norm_imbal = min((imbalance - 1) / 9.0, 1.0) if imbalance >= 1 else 0
        
        # 3. Correlation is already between 0 and 1
        norm_corr = min(correlation, 1.0)
        
        # Total score between 0 and 1 (weighted sum)
        weights = {'fsm': 0.5, 'imbal': 0.3, 'corr': 0.2}
        score = (weights['fsm'] * norm_fsm) + (weights['imbal'] * norm_imbal) + (weights['corr'] * norm_corr)
        
        # Scale to 0-100
        score_100 = score * 100
        
        # Categorize
        if score_100 < 30:
            category = "Easy"
        elif score_100 < 60:
            category = "Medium"
        else:
            category = "Hard"
            
        self.logger.info(f"Difficulty Score: {score_100:.2f} ({category})")
        return score_100, category
