import os
import logging
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.data_preprocessing.preprocess import DataPreprocessor
from src.meta_features.extractor import MetaFeatureExtractor
from src.models.classification import get_classification_models
from src.evaluation.metrics import evaluate_model
from src.utils.helpers import update_meta_dataset
from src.meta_learning.neural_mlp import MetaLearnerMLP
from src.difficulty.difficulty_score import DifficultyScorer
from src.hyperparameter.sensitivity import run_sensitivity_analysis
from src.automl_comparison.automl import compare_with_automl
from src.recommendation.recommender import Recommender
from src.visualization.dashboard import plot_performance_comparison, plot_predicted_vs_actual, plot_dataset_difficulty

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dummy_dataset(filename="data/raw/dummy.csv"):
    """Creates a dummy dataset for testing the pipeline."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        logger.info("Generating dummy dataset...")
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        df['target'] = y
        df.to_csv(filename, index=False)

def main():
    logger.info("Starting Meta-Learning Performance Analyzer Pipeline...")
    
    # Setup dummy data if it doesn't exist
    raw_data_path = "data/raw/dummy.csv"
    create_dummy_dataset(raw_data_path)
    
    # 1. Load & 2. Preprocess Dataset
    logger.info("--- Step 1 & 2: Loading and Preprocessing ---")
    preprocessor = DataPreprocessor(target_column='target')
    df_processed = preprocessor.process(raw_data_path)
    
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Extract Meta-features
    logger.info("--- Step 3: Meta-Feature Extraction ---")
    extractor = MetaFeatureExtractor(target_column='target')
    meta_features = extractor.extract_features(df_processed)
    
    # 4. Train Base Models & 5. Evaluate Performance & 6. Update Meta-Dataset
    logger.info("--- Steps 4, 5, 6: Training, Evaluation, Meta-Dataset Creation ---")
    classification_models = get_classification_models()
    actual_performances = {}
    
    for model_name, model_class in classification_models.items():
        model = model_class(random_state=42) if 'random_state' in model_class().get_params() else model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, task_type='classification')
        actual_performances[model_name] = metrics['accuracy']
        
        update_meta_dataset(
            dataset_name="dummy_dataset",
            meta_features=meta_features,
            model_name=model_name,
            performance_metrics=metrics,
            output_file="data/meta_data/meta_dataset.csv"
        )
        
    # Meta-Learning requires historical data (meta-dataset).
    # Since we only have one run, training a robust meta-learner here is illustrative.
    logger.info("--- Step 7: Train Meta-Learning Models ---")
    meta_df = pd.read_csv("data/meta_data/meta_dataset.csv")
    
    # Meta features used for learning
    feature_cols = [col for col in meta_features.keys()]
    X_meta = meta_df[feature_cols]
    y_meta_perf = meta_df['accuracy'] # Predicting accuracy
    
    meta_learner = MetaLearnerMLP(hidden_layer_sizes=(16, 8), max_iter=200)
    meta_learner.train(X_meta, y_meta_perf)
    
    # 8. Predict performance for the "new" dataset (we just reuse meta_features for demo)
    logger.info("--- Step 8: Predict Performance ---")
    # Duplicate meta-features shape for each candidate model if we included model_name encoding.
    # In this simple implementation, we predict the performance based purely on dataset features.
    predicted_perf = meta_learner.predict(pd.DataFrame([meta_features] * len(classification_models)))
    
    predicted_performances = {
        name: float(pred) for name, pred in zip(classification_models.keys(), predicted_perf)
    }
    logger.info(f"Predicted Accuracies: {predicted_performances}")
    
    # 9. Recommend Best Model
    logger.info("--- Step 9: Recommend Best Model ---")
    recommender = Recommender(metric_to_optimize='accuracy')
    ranked_models, best_model = recommender.recommend(predicted_performances)
    
    # Advanced Research Components
    logger.info("--- Advanced Research execution ---")
    scorer = DifficultyScorer()
    diff_score, diff_cat = scorer.compute_score(meta_features)
    
    # Sensitivity Analysis on Random Forest
    rf_model = classification_models['RandomForest'](random_state=42)
    run_sensitivity_analysis(rf_model, [10, 50, 100], X_train, y_train, 'n_estimators')
    
    # AutoML Comparison
    compare_with_automl(X_train, y_train, X_test, y_test, system_predicted_accuracy=predicted_performances[best_model], dataset_name="dummy_dataset")
    
    # 10. Visualize Results
    logger.info("--- Step 10: Visualize Results ---")
    plot_performance_comparison(actual_performances)
    
    # For actual vs pred scatter, we compare the arrays directly
    plot_predicted_vs_actual(list(actual_performances.values()), list(predicted_performances.values()))
    
    plot_dataset_difficulty([diff_score], ["dummy_dataset"])
    
    logger.info("Pipeline execution completed successfully. Check 'outputs/plots' for visualizations.")

if __name__ == "__main__":
    main()
