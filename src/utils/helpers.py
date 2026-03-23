"""Utility functions for customer segmentation analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    
    # Set additional random seeds if libraries are available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Random seeds set to {seed}")


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save analysis results to file.
    
    Args:
        results: Results dictionary to save.
        filepath: Path to save the results.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = _convert_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load analysis results from file.
    
    Args:
        filepath: Path to load the results from.
        
    Returns:
        Loaded results dictionary.
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results loaded from {filepath}")
    return results


def _convert_for_json(obj: Any) -> Any:
    """Convert objects for JSON serialization.
    
    Args:
        obj: Object to convert.
        
    Returns:
        JSON-serializable object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj


def calculate_statistics(df: pd.DataFrame, groupby_col: str = None) -> Dict[str, Any]:
    """Calculate descriptive statistics for a DataFrame.
    
    Args:
        df: Input DataFrame.
        groupby_col: Column to group by (optional).
        
    Returns:
        Dictionary with statistics.
    """
    if groupby_col and groupby_col in df.columns:
        stats = df.groupby(groupby_col).describe()
    else:
        stats = df.describe()
    
    return {
        'statistics': stats,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns.
    
    Args:
        df: Input DataFrame.
        required_columns: List of required column names.
        
    Returns:
        True if validation passes, False otherwise.
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    logger.info("Data validation passed")
    return True


def create_summary_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    model_name: str,
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """Create a summary report for segmentation results.
    
    Args:
        df: Customer DataFrame.
        labels: Cluster labels.
        model_name: Name of the model used.
        metrics: Evaluation metrics.
        
    Returns:
        Summary report dictionary.
    """
    # Basic statistics
    n_customers = len(df)
    n_clusters = len(np.unique(labels))
    n_noise = np.sum(labels == -1) if -1 in labels else 0
    
    # Cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    # Feature statistics by cluster
    feature_stats = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_data = df[labels == cluster_id]
        feature_stats[f'cluster_{cluster_id}'] = {
            col: {
                'mean': cluster_data[col].mean(),
                'std': cluster_data[col].std(),
                'median': cluster_data[col].median()
            }
            for col in numeric_columns
        }
    
    report = {
        'model_name': model_name,
        'n_customers': n_customers,
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'cluster_sizes': cluster_sizes,
        'evaluation_metrics': metrics,
        'feature_statistics': feature_stats,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info(f"Summary report created for {model_name}")
    return report


def format_currency(value: float, currency: str = 'USD') -> str:
    """Format a numeric value as currency.
    
    Args:
        value: Numeric value to format.
        currency: Currency code.
        
    Returns:
        Formatted currency string.
    """
    if currency == 'USD':
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a numeric value as percentage.
    
    Args:
        value: Numeric value to format.
        decimals: Number of decimal places.
        
    Returns:
        Formatted percentage string.
    """
    return f"{value:.{decimals}f}%"


def create_feature_importance_plot(
    feature_names: List[str],
    importance_scores: List[float],
    title: str = "Feature Importance"
) -> Dict[str, Any]:
    """Create feature importance visualization data.
    
    Args:
        feature_names: List of feature names.
        importance_scores: List of importance scores.
        title: Plot title.
        
    Returns:
        Dictionary with plot data.
    """
    # Sort by importance
    sorted_data = sorted(zip(feature_names, importance_scores), 
                        key=lambda x: x[1], reverse=True)
    
    sorted_features, sorted_scores = zip(*sorted_data)
    
    return {
        'title': title,
        'features': list(sorted_features),
        'scores': list(sorted_scores),
        'max_score': max(importance_scores),
        'min_score': min(importance_scores)
    }


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame.
        column: Column to analyze.
        multiplier: IQR multiplier for outlier detection.
        
    Returns:
        Boolean Series indicating outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def create_cluster_summary_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: List[str]
) -> pd.DataFrame:
    """Create a summary table for cluster characteristics.
    
    Args:
        df: Customer DataFrame.
        labels: Cluster labels.
        feature_columns: Features to include in summary.
        
    Returns:
        Summary DataFrame.
    """
    summary_data = []
    
    for cluster_id in sorted(np.unique(labels)):
        if cluster_id == -1:  # Skip noise points
            continue
            
        cluster_data = df[labels == cluster_id]
        
        row = {'Cluster': cluster_id}
        row['Size'] = len(cluster_data)
        row['Size_Percentage'] = len(cluster_data) / len(df) * 100
        
        # Add feature statistics
        for feature in feature_columns:
            if feature in cluster_data.columns:
                row[f'{feature}_mean'] = cluster_data[feature].mean()
                row[f'{feature}_std'] = cluster_data[feature].std()
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)
