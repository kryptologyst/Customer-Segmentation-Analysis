"""Data generation and preprocessing module for customer segmentation analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_blobs
import logging

logger = logging.getLogger(__name__)


class CustomerDataGenerator:
    """Generate synthetic customer data for segmentation analysis."""
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_customer_data(
        self,
        n_customers: int = 1000,
        n_features: int = 5,
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """Generate synthetic customer data.
        
        Args:
            n_customers: Number of customers to generate.
            n_features: Number of features per customer.
            noise_level: Amount of noise to add to the data.
            
        Returns:
            DataFrame with customer features.
        """
        logger.info(f"Generating {n_customers} customers with {n_features} features")
        
        # Generate base clusters using make_blobs
        X, y = make_blobs(
            n_samples=n_customers,
            centers=5,  # Natural customer segments
            n_features=n_features,
            cluster_std=2.0,
            random_state=self.random_seed
        )
        
        # Add noise
        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise
        
        # Create feature names
        feature_names = [
            'age', 'annual_income', 'spending_score', 
            'purchase_frequency', 'avg_order_value'
        ]
        
        # Scale features to realistic ranges
        X_scaled = self._scale_to_realistic_ranges(X, feature_names)
        
        # Create DataFrame
        df = pd.DataFrame(X_scaled, columns=feature_names)
        df['customer_id'] = range(1, n_customers + 1)
        df['true_segment'] = y
        
        # Add some additional derived features
        df['income_spending_ratio'] = df['annual_income'] / (df['spending_score'] + 1)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                                labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        logger.info(f"Generated data shape: {df.shape}")
        return df
    
    def _scale_to_realistic_ranges(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Scale features to realistic customer data ranges.
        
        Args:
            X: Raw feature matrix.
            feature_names: Names of the features.
            
        Returns:
            Scaled feature matrix.
        """
        X_scaled = X.copy()
        
        # Define realistic ranges for each feature
        ranges = {
            'age': (18, 80),
            'annual_income': (20, 150),  # in thousands
            'spending_score': (1, 100),
            'purchase_frequency': (1, 50),  # purchases per year
            'avg_order_value': (10, 500)  # in dollars
        }
        
        for i, feature in enumerate(feature_names):
            if feature in ranges:
                min_val, max_val = ranges[feature]
                # Scale from [min, max] to [0, 1] then to [min_val, max_val]
                X_scaled[:, i] = (X_scaled[:, i] - X_scaled[:, i].min()) / (X_scaled[:, i].max() - X_scaled[:, i].min())
                X_scaled[:, i] = X_scaled[:, i] * (max_val - min_val) + min_val
        
        return X_scaled


class DataPreprocessor:
    """Preprocess customer data for segmentation analysis."""
    
    def __init__(self, scaler_type: str = 'standard') -> None:
        """Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler to use ('standard' or 'minmax').
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        self.feature_columns: Optional[List[str]] = None
        
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Fit scaler and transform data.
        
        Args:
            df: Input DataFrame.
            feature_columns: Columns to use for scaling.
            
        Returns:
            Scaled feature matrix.
        """
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Scaled {len(feature_columns)} features using {type(self.scaler).__name__}")
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Scaled feature matrix.
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted first")
        
        X = df[self.feature_columns].values
        return self.scaler.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data.
        
        Args:
            X_scaled: Scaled feature matrix.
            
        Returns:
            Original scale feature matrix.
        """
        return self.scaler.inverse_transform(X_scaled)


def load_sample_data() -> pd.DataFrame:
    """Load the original sample data for backward compatibility.
    
    Returns:
        DataFrame with original sample data.
    """
    data = {
        'Age': [19, 35, 26, 27, 19, 27, 27, 32, 25, 35],
        'Annual_Income_k$': [15, 35, 35, 19, 27, 75, 45, 40, 60, 50],
        'Spending_Score': [39, 81, 6, 77, 40, 76, 20, 8, 40, 35]
    }
    
    df = pd.DataFrame(data)
    df['customer_id'] = range(1, len(df) + 1)
    
    logger.info(f"Loaded sample data with {len(df)} customers")
    return df
