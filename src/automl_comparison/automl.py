import logging
import pandas as pd
from sklearn.metrics import accuracy_score
import time

logger = logging.getLogger(__name__)

# Attempt to load TPOT or Auto-Sklearn
try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

# Auto-Sklearn usually requires Linux environment. We degrade gracefully if not installed.

def compare_with_automl(X_train, y_train, X_test, y_test, system_predicted_accuracy, dataset_name="Dataset"):
    """
    Compare the accuracy recommended by our meta-learning system with a true AutoML baseline (e.g. TPOT).
    
    Args:
        X_train, y_train, X_test, y_test: Train/test splits.
        system_predicted_accuracy (float): The accuracy predicted/attained by our top recommended model.
        dataset_name (str): Label for logs.
        
    Returns:
        pd.DataFrame: Comparison table.
    """
    logger.info(f"Starting AutoML comparison for {dataset_name}...")
    
    results = {
        'System': ['Meta-Learning Analyzer'],
        'Accuracy': [system_predicted_accuracy],
        'Time_Seconds': [0.0] # Placeholder, typically fast due to meta-learning lookup
    }
    
    if TPOT_AVAILABLE:
        logger.info("TPOT AutoML is available. Running baseline computation (max 2 mins).")
        # Limited to 2 minutes for quick comparison
        tpot = TPOTClassifier(max_time_mins=2, population_size=20, random_state=42, verbosity=2)
        
        start_time = time.time()
        tpot.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        y_pred = tpot.predict(X_test)
        automl_acc = accuracy_score(y_test, y_pred)
        
        results['System'].append('TPOT AutoML')
        results['Accuracy'].append(automl_acc)
        results['Time_Seconds'].append(elapsed_time)
    else:
        logger.warning("TPOT not available. AutoML comparison skipped. Add auto-sklearn or tpot to requirements.")
        
    comparison_df = pd.DataFrame(results)
    logger.info(f"AutoML Comparison DataFrame:\n{comparison_df}")
    
    return comparison_df
