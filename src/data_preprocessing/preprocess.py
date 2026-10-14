import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class DataPreprocessor:
    """
    Class to load, clean, and preprocess datasets.
    Handles missing values, encodes categorical variables, and scales numerical features.
    """
    def __init__(self, target_column=None):
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def load_data(self, filepath):
        """Load CSV dataset from a given path."""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded dataset from {filepath} with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            raise e

    def clean_data(self, df):
        """
        Handle missing values:
        - Mean for numerical
        - Mode for categorical
        """
        self.logger.info("Cleaning data (handling missing values)...")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def encode_data(self, df):
        """
        Encode categorical variables using LabelEncoder.
        """
        self.logger.info("Encoding categorical columns...")
        for col in df.select_dtypes(include=['object', 'category']).columns:
            # Simple label encoding (can be extended to OneHot encoding if required)
            le = LabelEncoder()
            # Handle potential nan (though clean_data should fix this, defensive programming)
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def scale_data(self, df):
        """
        Normalize numerical features using StandardScaler.
        Excludes target column from scaling if provided.
        """
        self.logger.info("Scaling numerical features...")
        num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        if self.target_column and self.target_column in num_cols:
            num_cols.remove(self.target_column)

        if len(num_cols) > 0:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        
        return df

    def process(self, filepath):
        """
        Executes the entire preprocessing pipeline.
        Returns the processed DataFrame.
        """
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.encode_data(df)
        df = self.scale_data(df)
        self.logger.info("Data preprocessing completed successfully.")
        return df
