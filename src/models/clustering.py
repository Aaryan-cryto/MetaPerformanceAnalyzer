import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

def train_kmeans(X_train, n_clusters=3, **kwargs):
    """Train a KMeans clustering model."""
    logger.info(f"Training KMeans models with {n_clusters} clusters...")
    model = KMeans(n_clusters=n_clusters, **kwargs)
    model.fit(X_train)
    return model

def get_clustering_models():
    """Return a dictionary of uninitialized clustering models."""
    return {
        'KMeans': KMeans
    }
