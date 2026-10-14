import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

logger = logging.getLogger(__name__)

def plot_performance_comparison(models_performance, output_dir="outputs/plots", filename="model_comparison.png"):
    """
    Plot bar chart comparing model performances.
    """
    logger.info("Plotting performance comparison...")
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(models_performance.keys())
    scores = list(models_performance.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=scores, palette="viridis")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")

def plot_predicted_vs_actual(y_true, y_pred, output_dir="outputs/plots", filename="pred_vs_actual.png"):
    """
    Plot scatter of predicted vs actual meta-learning performance.
    """
    logger.info("Plotting predicted vs actual performance...")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # Plot diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title("Meta-Learning: Predicted vs Actual Performance")
    plt.xlabel("Actual Performance")
    plt.ylabel("Predicted Performance")
    plt.grid(True)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")

def plot_dataset_difficulty(difficulty_scores, datasets, output_dir="outputs/plots", filename="difficulty_dist.png"):
    """
    Plot difficulty scores across datasets.
    """
    logger.info("Plotting dataset difficulty...")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=datasets, y=difficulty_scores, palette="magma")
    plt.title("Dataset Difficulty Scores")
    plt.ylabel("Difficulty (0-100)")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45, ha='right')
    # Add horizontal lines for categories (e.g., 30=Easy, 60=Medium)
    plt.axhline(30, color='green', linestyle='--', label='Easy Threshold')
    plt.axhline(60, color='orange', linestyle='--', label='Medium Threshold')
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved plot to {save_path}")
