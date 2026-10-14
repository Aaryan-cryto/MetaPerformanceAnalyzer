import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
import logging

class MetaFeatureExtractor:
    """
    Extracts meta-features from a given dataset to be used for meta-learning.
    """
    def __init__(self, target_column=None):
        self.target_column = target_column
        self.logger = logging.getLogger(__name__)

    def extract_features(self, df):
        """
        Extract descriptive statistical, structural, and information-theoretic meta-features.
        Returns a dictionary of meta-features.
        """
        self.logger.info("Starting meta-feature extraction...")
        meta_features = {}

        # Separation of features and target optionally
        X = df.drop(columns=[self.target_column]) if self.target_column else df
        y = df[self.target_column] if self.target_column else None

        # 1. Structural features
        meta_features['n_samples'] = X.shape[0]
        meta_features['n_features'] = X.shape[1]
        
        if meta_features['n_samples'] > 0:
            meta_features['feature_to_sample_ratio'] = meta_features['n_features'] / meta_features['n_samples']
        else:
            meta_features['feature_to_sample_ratio'] = 0.0

        # 2. Statistical features (Aggregate across numerical features)
        numeric_X = X.select_dtypes(include=[np.number])
        if not numeric_X.empty:
            meta_features['mean_variance'] = numeric_X.var().mean()
            meta_features['mean_skewness'] = numeric_X.apply(skew).mean()
            
            # 3. Correlation-based: Average absolute pairwise correlation
            if numeric_X.shape[1] > 1:
                corr_matrix = numeric_X.corr().abs()
                # upper triangle without diagonal
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                meta_features['mean_correlation'] = upper_tri.mean().mean() # average of average correlations
            else:
                meta_features['mean_correlation'] = 0.0
        else:
            meta_features['mean_variance'] = 0.0
            meta_features['mean_skewness'] = 0.0
            meta_features['mean_correlation'] = 0.0

        # 4. Information-theoretic (Target related)
        if y is not None:
            # Compute entropy for classification targets
            if len(y.unique()) < 20: # Assuming it's a classification classification roughly
                val_counts = y.value_counts(normalize=True)
                meta_features['target_entropy'] = entropy(val_counts)
                
                # Class imbalance ratio (Majority class size / Minority class size)
                min_class_ratio = val_counts.min()
                max_class_ratio = val_counts.max()
                meta_features['class_imbalance_ratio'] = max_class_ratio / min_class_ratio if min_class_ratio > 0 else 0
            else:
                meta_features['target_entropy'] = 0.0 # Standardize return if continuous
                meta_features['class_imbalance_ratio'] = 1.0
        else:
            meta_features['target_entropy'] = np.nan
            meta_features['class_imbalance_ratio'] = np.nan

        self.logger.info("Extraction completed successfully.")
        return meta_features
