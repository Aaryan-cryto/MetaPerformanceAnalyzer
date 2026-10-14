import logging
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os

logger = logging.getLogger(__name__)

def run_sensitivity_analysis(model, param_grid, X, y, param_name, output_dir="outputs/plots"):
    """
    Run hyperparameter sensitivity analysis by varying a single parameter.
    
    Args:
        model: Instantiate sklearn model (e.g., RandomForestClassifier()).
        param_grid: List of values for the hyperparameter.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target array.
        param_name (str): Name of the hyperparameter being varied.
        output_dir (str): Directory to save the plot.
    """
    logger.info(f"Running sensitivity analysis for {param_name}...")
    
    scores_mean = []
    scores_std = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track performance across different values of param_name
    for val in param_grid:
        logger.info(f"Evaluating {param_name}={val}...")
        model.set_params(**{param_name: val})
        
        # Using cross_val_score for robust evaluation
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            scores_mean.append(cv_scores.mean())
            scores_std.append(cv_scores.std())
        except Exception as e:
            logger.error(f"Error evaluating {param_name}={val}: {str(e)}")
            scores_mean.append(0)
            scores_std.append(0)
            
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid, scores_mean, marker='o', label='Mean Accuracy')
    plt.fill_between(
        param_grid,
        [m - s for m, s in zip(scores_mean, scores_std)],
        [m + s for m, s in zip(scores_mean, scores_std)],
        alpha=0.2, color='b', label='±1 Std Dev'
    )
    plt.title(f"Hyperparameter Sensitivity: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"sensitivity_{param_name}.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Sensitivity plot saved to {plot_path}")
    
    # Return as a dataframe table
    results_df = pd.DataFrame({
        param_name: param_grid,
        'mean_accuracy': scores_mean,
        'std_accuracy': scores_std
    })
    return results_df
