import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def update_meta_dataset(dataset_name, meta_features, model_name, performance_metrics, output_file="data/meta_data/meta_dataset.csv"):
    """
    Appends a new record to the meta-dataset CSV file.
    
    Args:
        dataset_name (str): Name of the dataset.
        meta_features (dict): Dictionary of extracted meta-features.
        model_name (str): Name of the evaluated model.
        performance_metrics (dict): Dictionary of performance metrics.
        output_file (str): Path to the CSV file.
    """
    logger.info(f"Updating meta-dataset for {dataset_name} run on {model_name}...")
    
    # Create the row data
    row_data = {'dataset_name': dataset_name, 'model_name': model_name}
    row_data.update(meta_features)
    row_data.update(performance_metrics)
    
    df_new = pd.DataFrame([row_data])
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(output_file):
        df_new.to_csv(output_file, index=False)
        logger.info(f"Created new meta-dataset at {output_file}.")
    else:
        # Load existing to prevent exact duplicates (optional heuristic)
        df_existing = pd.read_csv(output_file)
        
        # simple check: if we already ran this model on this dataset, we can skip or overwrite
        # For simplicity, we just append here.
        df_new.to_csv(output_file, mode='a', header=False, index=False)
        logger.info(f"Appended to existing meta-dataset at {output_file}.")
