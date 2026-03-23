"""Main analysis script for customer segmentation."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.generator import CustomerDataGenerator, DataPreprocessor, load_sample_data
from models.segmentation import (
    KMeansSegmentation, 
    GaussianMixtureSegmentation, 
    HDBSCANSegmentation,
    SegmentationEvaluator,
    find_optimal_k
)
from eval.business_metrics import BusinessMetricsCalculator, SegmentationInsights
from viz.plots import SegmentationVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomerSegmentationAnalysis:
    """Main class for customer segmentation analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """Initialize the analysis.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = self._load_config(config_path)
        self.data_generator = CustomerDataGenerator(
            random_seed=self.config.data.synthetic.random_seed
        )
        self.preprocessor = DataPreprocessor()
        self.evaluator = SegmentationEvaluator()
        self.business_calculator = BusinessMetricsCalculator()
        self.insights_generator = SegmentationInsights()
        self.visualizer = SegmentationVisualizer(
            figsize=self.config.visualization.figure_size,
            dpi=self.config.visualization.dpi
        )
        
        # Results storage
        self.df: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.models: Dict[str, any] = {}
        self.results: Dict[str, any] = {}
        
    def _load_config(self, config_path: str) -> OmegaConf:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file.
            
        Returns:
            Configuration object.
        """
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
        else:
            # Default configuration
            config = OmegaConf.create({
                'data': {
                    'synthetic': {
                        'n_customers': 1000,
                        'n_features': 5,
                        'random_seed': 42,
                        'noise_level': 0.1
                    },
                    'features': [
                        'age', 'annual_income', 'spending_score', 
                        'purchase_frequency', 'avg_order_value'
                    ]
                },
                'models': {
                    'kmeans': {'n_clusters': 5, 'random_state': 42},
                    'gmm': {'n_components': 5, 'random_state': 42},
                    'hdbscan': {'min_cluster_size': 20, 'min_samples': 10}
                },
                'visualization': {
                    'figure_size': [12, 8],
                    'dpi': 300
                }
            })
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def generate_data(self, use_sample: bool = False) -> pd.DataFrame:
        """Generate or load customer data.
        
        Args:
            use_sample: Whether to use the original sample data.
            
        Returns:
            Customer DataFrame.
        """
        if use_sample:
            self.df = load_sample_data()
            # Rename columns to match expected format
            self.df = self.df.rename(columns={
                'Age': 'age',
                'Annual_Income_k$': 'annual_income',
                'Spending_Score': 'spending_score'
            })
        else:
            self.df = self.data_generator.generate_customer_data(
                n_customers=self.config.data.synthetic.n_customers,
                n_features=self.config.data.synthetic.n_features,
                noise_level=self.config.data.synthetic.noise_level
            )
        
        logger.info(f"Generated data with {len(self.df)} customers and {len(self.df.columns)} features")
        return self.df
    
    def preprocess_data(self) -> np.ndarray:
        """Preprocess the data for clustering.
        
        Returns:
            Scaled feature matrix.
        """
        if self.df is None:
            raise ValueError("Data must be generated first")
        
        feature_columns = self.config.data.features
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        if not available_features:
            raise ValueError(f"No features found from {feature_columns}")
        
        self.X_scaled = self.preprocessor.fit_transform(self.df, available_features)
        
        logger.info(f"Preprocessed data: {self.X_scaled.shape}")
        return self.X_scaled
    
    def find_optimal_clusters(self) -> Tuple[int, Dict[str, List[float]]]:
        """Find optimal number of clusters.
        
        Returns:
            Tuple of (optimal_k, metrics_dict).
        """
        if self.X_scaled is None:
            raise ValueError("Data must be preprocessed first")
        
        optimal_k, metrics = find_optimal_k(self.X_scaled, max_k=10)
        
        # Update config with optimal k
        self.config.models.kmeans.n_clusters = optimal_k
        self.config.models.gmm.n_components = optimal_k
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k, metrics
    
    def train_models(self) -> Dict[str, any]:
        """Train all segmentation models.
        
        Returns:
            Dictionary of trained models.
        """
        if self.X_scaled is None:
            raise ValueError("Data must be preprocessed first")
        
        # Initialize models
        models = {
            'kmeans': KMeansSegmentation(**self.config.models.kmeans),
            'gmm': GaussianMixtureSegmentation(**self.config.models.gmm),
            'hdbscan': HDBSCANSegmentation(**self.config.models.hdbscan)
        }
        
        # Train models
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            model.fit(self.X_scaled)
            self.evaluator.evaluate_model(
                model, 
                self.X_scaled, 
                self.df.get('true_segment') if 'true_segment' in self.df.columns else None
            )
        
        self.models = models
        logger.info("All models trained successfully")
        return models
    
    def generate_business_insights(self) -> Dict[str, any]:
        """Generate business insights and recommendations.
        
        Returns:
            Dictionary with business insights.
        """
        if not self.models:
            raise ValueError("Models must be trained first")
        
        insights = {}
        
        # Use K-Means results for business analysis (most stable)
        kmeans_model = self.models['kmeans']
        labels = kmeans_model.labels_
        
        # Calculate cluster profiles
        feature_columns = [col for col in self.config.data.features if col in self.df.columns]
        profiles_df = self.business_calculator.calculate_cluster_profiles(
            self.df, labels, feature_columns
        )
        
        # Calculate CLV
        clv_df = self.business_calculator.calculate_customer_lifetime_value(
            self.df, labels
        )
        
        # Generate recommendations
        recommendations = self.insights_generator.generate_recommendations(
            profiles_df, clv_df
        )
        
        # Identify opportunities
        opportunities = self.insights_generator.identify_opportunities(profiles_df)
        
        insights = {
            'profiles': profiles_df,
            'clv': clv_df,
            'recommendations': recommendations,
            'opportunities': opportunities,
            'leaderboard': self.evaluator.get_leaderboard()
        }
        
        self.results = insights
        logger.info("Business insights generated successfully")
        return insights
    
    def create_visualizations(self, save_dir: str = "assets") -> Dict[str, str]:
        """Create all visualizations.
        
        Args:
            save_dir: Directory to save visualizations.
            
        Returns:
            Dictionary with saved plot paths.
        """
        if not self.models or not self.results:
            raise ValueError("Models and results must be available")
        
        os.makedirs(save_dir, exist_ok=True)
        plot_paths = {}
        
        # Use K-Means results for visualization
        kmeans_model = self.models['kmeans']
        labels = kmeans_model.labels_
        
        # Cluster visualization
        feature_columns = [col for col in self.config.data.features if col in self.df.columns]
        fig1 = self.visualizer.plot_clusters_2d(
            self.df, labels, feature_columns,
            save_path=os.path.join(save_dir, "clusters_2d.png")
        )
        plot_paths['clusters_2d'] = os.path.join(save_dir, "clusters_2d.png")
        
        # Cluster profiles
        fig2 = self.visualizer.plot_cluster_profiles(
            self.results['profiles'],
            save_path=os.path.join(save_dir, "cluster_profiles.png")
        )
        plot_paths['cluster_profiles'] = os.path.join(save_dir, "cluster_profiles.png")
        
        # CLV analysis
        fig3 = self.visualizer.plot_clv_analysis(
            self.results['clv'],
            save_path=os.path.join(save_dir, "clv_analysis.png")
        )
        plot_paths['clv_analysis'] = os.path.join(save_dir, "clv_analysis.png")
        
        # Interactive dashboard
        fig4 = self.visualizer.create_interactive_dashboard(
            self.df, labels, 
            self.results['profiles'], 
            self.results['clv']
        )
        fig4.write_html(os.path.join(save_dir, "interactive_dashboard.html"))
        plot_paths['interactive_dashboard'] = os.path.join(save_dir, "interactive_dashboard.html")
        
        logger.info(f"Visualizations saved to {save_dir}")
        return plot_paths
    
    def run_complete_analysis(self, use_sample: bool = False) -> Dict[str, any]:
        """Run the complete analysis pipeline.
        
        Args:
            use_sample: Whether to use sample data.
            
        Returns:
            Complete analysis results.
        """
        logger.info("Starting complete customer segmentation analysis...")
        
        # Generate data
        self.generate_data(use_sample=use_sample)
        
        # Preprocess data
        self.preprocess_data()
        
        # Find optimal clusters
        optimal_k, elbow_metrics = self.find_optimal_clusters()
        
        # Train models
        self.train_models()
        
        # Generate business insights
        self.generate_business_insights()
        
        # Create visualizations
        plot_paths = self.create_visualizations()
        
        # Compile results
        complete_results = {
            'data': self.df,
            'optimal_k': optimal_k,
            'elbow_metrics': elbow_metrics,
            'models': self.models,
            'business_insights': self.results,
            'plot_paths': plot_paths,
            'config': self.config
        }
        
        logger.info("Complete analysis finished successfully")
        return complete_results
    
    def print_summary(self) -> None:
        """Print analysis summary."""
        if not self.results:
            logger.warning("No results available. Run analysis first.")
            return
        
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION ANALYSIS SUMMARY")
        print("="*60)
        
        # Model performance
        print("\nMODEL PERFORMANCE:")
        print("-" * 30)
        leaderboard = self.results['leaderboard']
        if not leaderboard.empty:
            for _, row in leaderboard.iterrows():
                print(f"{row['model']}:")
                print(f"  - Silhouette Score: {row.get('silhouette_score', 'N/A'):.3f}")
                print(f"  - Number of Clusters: {row.get('n_clusters', 'N/A')}")
                print()
        
        # Cluster profiles
        print("CLUSTER PROFILES:")
        print("-" * 30)
        profiles = self.results['profiles']
        for _, profile in profiles.iterrows():
            print(f"{profile['cluster_name']} (Cluster {profile['cluster_id']}):")
            print(f"  - Size: {profile['size']} customers ({profile['size_pct']:.1f}%)")
            if 'avg_income' in profile:
                print(f"  - Avg Income: ${profile['avg_income']:.0f}k")
            if 'avg_spending' in profile:
                print(f"  - Avg Spending Score: {profile['avg_spending']:.0f}")
            print()
        
        # Business opportunities
        print("KEY OPPORTUNITIES:")
        print("-" * 30)
        for opportunity in self.results['opportunities']:
            print(f"• {opportunity}")
        
        print("\n" + "="*60)


def main():
    """Main function to run the analysis."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Initialize analysis
    analysis = CustomerSegmentationAnalysis()
    
    # Run complete analysis
    results = analysis.run_complete_analysis(use_sample=False)
    
    # Print summary
    analysis.print_summary()
    
    # Save results
    import pickle
    with open('assets/analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logger.info("Analysis complete! Results saved to assets/analysis_results.pkl")


if __name__ == "__main__":
    main()
