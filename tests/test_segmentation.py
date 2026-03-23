"""Tests for customer segmentation analysis."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import CustomerDataGenerator, DataPreprocessor
from models.segmentation import KMeansSegmentation, GaussianMixtureSegmentation, HDBSCANSegmentation
from eval.business_metrics import BusinessMetricsCalculator
from utils.helpers import set_random_seeds, validate_data


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_data_generator_init(self):
        """Test data generator initialization."""
        generator = CustomerDataGenerator(random_seed=42)
        assert generator.random_seed == 42
    
    def test_generate_customer_data(self):
        """Test customer data generation."""
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_customer_data(n_customers=100)
        
        assert len(df) == 100
        assert 'customer_id' in df.columns
        assert 'true_segment' in df.columns
        assert 'age' in df.columns
        assert 'annual_income' in df.columns
        assert 'spending_score' in df.columns
    
    def test_data_ranges(self):
        """Test that generated data is within expected ranges."""
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_customer_data(n_customers=1000)
        
        assert df['age'].min() >= 18
        assert df['age'].max() <= 80
        assert df['annual_income'].min() >= 20
        assert df['annual_income'].max() <= 150
        assert df['spending_score'].min() >= 1
        assert df['spending_score'].max() <= 100


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(scaler_type='standard')
        assert preprocessor.scaler is not None
        
        preprocessor = DataPreprocessor(scaler_type='minmax')
        assert preprocessor.scaler is not None
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_customer_data(n_customers=100)
        
        preprocessor = DataPreprocessor()
        feature_columns = ['age', 'annual_income', 'spending_score']
        X_scaled = preprocessor.fit_transform(df, feature_columns)
        
        assert X_scaled.shape == (100, 3)
        assert preprocessor.feature_columns == feature_columns
    
    def test_transform(self):
        """Test transform functionality."""
        generator = CustomerDataGenerator(random_seed=42)
        df = generator.generate_customer_data(n_customers=100)
        
        preprocessor = DataPreprocessor()
        feature_columns = ['age', 'annual_income', 'spending_score']
        preprocessor.fit_transform(df, feature_columns)
        
        # Test transform on same data
        X_scaled = preprocessor.transform(df)
        assert X_scaled.shape == (100, 3)


class TestSegmentationModels:
    """Test segmentation models."""
    
    def setup_method(self):
        """Set up test data."""
        generator = CustomerDataGenerator(random_seed=42)
        self.df = generator.generate_customer_data(n_customers=100)
        
        preprocessor = DataPreprocessor()
        feature_columns = ['age', 'annual_income', 'spending_score']
        self.X_scaled = preprocessor.fit_transform(self.df, feature_columns)
    
    def test_kmeans_segmentation(self):
        """Test K-Means segmentation."""
        model = KMeansSegmentation(n_clusters=3, random_state=42)
        model.fit(self.X_scaled)
        
        assert model.labels_ is not None
        assert len(model.labels_) == len(self.X_scaled)
        assert model.n_clusters_ == 3
    
    def test_gmm_segmentation(self):
        """Test Gaussian Mixture Model segmentation."""
        model = GaussianMixtureSegmentation(n_components=3, random_state=42)
        model.fit(self.X_scaled)
        
        assert model.labels_ is not None
        assert len(model.labels_) == len(self.X_scaled)
        assert model.n_clusters_ == 3
    
    def test_hdbscan_segmentation(self):
        """Test HDBSCAN segmentation."""
        model = HDBSCANSegmentation(min_cluster_size=5)
        model.fit(self.X_scaled)
        
        assert model.labels_ is not None
        assert len(model.labels_) == len(self.X_scaled)
        assert model.n_clusters_ >= 0  # HDBSCAN can find 0 clusters
    
    def test_model_predict(self):
        """Test model prediction functionality."""
        model = KMeansSegmentation(n_clusters=3, random_state=42)
        model.fit(self.X_scaled)
        
        predictions = model.predict(self.X_scaled)
        assert len(predictions) == len(self.X_scaled)
        assert np.array_equal(predictions, model.labels_)


class TestBusinessMetrics:
    """Test business metrics calculation."""
    
    def setup_method(self):
        """Set up test data."""
        generator = CustomerDataGenerator(random_seed=42)
        self.df = generator.generate_customer_data(n_customers=100)
        
        # Create simple labels
        self.labels = np.random.randint(0, 3, len(self.df))
    
    def test_cluster_profiles(self):
        """Test cluster profile calculation."""
        calculator = BusinessMetricsCalculator()
        feature_columns = ['age', 'annual_income', 'spending_score']
        
        profiles_df = calculator.calculate_cluster_profiles(
            self.df, self.labels, feature_columns
        )
        
        assert len(profiles_df) > 0
        assert 'cluster_id' in profiles_df.columns
        assert 'size' in profiles_df.columns
        assert 'cluster_name' in profiles_df.columns
    
    def test_clv_calculation(self):
        """Test CLV calculation."""
        calculator = BusinessMetricsCalculator()
        
        clv_df = calculator.calculate_customer_lifetime_value(
            self.df, self.labels
        )
        
        assert len(clv_df) > 0
        assert 'cluster_id' in clv_df.columns
        assert 'clv' in clv_df.columns
        assert 'size' in clv_df.columns


class TestUtilities:
    """Test utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        # This is hard to test directly, but we can check it doesn't raise errors
        assert True
    
    def test_validate_data(self):
        """Test data validation."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50, 60, 70],
            'spending': [40, 50, 60]
        })
        
        # Valid data
        assert validate_data(df, ['age', 'income']) == True
        
        # Invalid data
        assert validate_data(df, ['age', 'missing_column']) == False


if __name__ == "__main__":
    pytest.main([__file__])
