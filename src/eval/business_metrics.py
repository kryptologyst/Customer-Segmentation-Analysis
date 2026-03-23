"""Evaluation and business metrics module for customer segmentation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BusinessMetricsCalculator:
    """Calculate business-relevant metrics for customer segmentation."""
    
    def __init__(self) -> None:
        """Initialize the business metrics calculator."""
        self.metrics = {}
    
    def calculate_cluster_profiles(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Calculate cluster profiles with business metrics.
        
        Args:
            df: Customer DataFrame.
            labels: Cluster labels.
            feature_columns: Feature columns to analyze.
            
        Returns:
            DataFrame with cluster profiles.
        """
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = []
        
        for cluster_id in sorted(np.unique(labels)):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'size_pct': len(cluster_data) / len(df_with_clusters) * 100
            }
            
            # Feature statistics
            for feature in feature_columns:
                if feature in cluster_data.columns:
                    stats[f'{feature}_mean'] = cluster_data[feature].mean()
                    stats[f'{feature}_std'] = cluster_data[feature].std()
                    stats[f'{feature}_median'] = cluster_data[feature].median()
            
            # Business metrics
            if 'annual_income' in cluster_data.columns and 'spending_score' in cluster_data.columns:
                stats['avg_income'] = cluster_data['annual_income'].mean()
                stats['avg_spending'] = cluster_data['spending_score'].mean()
                stats['income_spending_ratio'] = stats['avg_income'] / (stats['avg_spending'] + 1)
            
            if 'purchase_frequency' in cluster_data.columns and 'avg_order_value' in cluster_data.columns:
                stats['avg_purchase_freq'] = cluster_data['purchase_frequency'].mean()
                stats['avg_order_value'] = cluster_data['avg_order_value'].mean()
                stats['estimated_annual_value'] = stats['avg_purchase_freq'] * stats['avg_order_value']
            
            cluster_stats.append(stats)
        
        profiles_df = pd.DataFrame(cluster_stats)
        
        # Add cluster names based on characteristics
        profiles_df['cluster_name'] = self._generate_cluster_names(profiles_df)
        
        logger.info(f"Generated profiles for {len(profiles_df)} clusters")
        return profiles_df
    
    def _generate_cluster_names(self, profiles_df: pd.DataFrame) -> List[str]:
        """Generate descriptive names for clusters.
        
        Args:
            profiles_df: Cluster profiles DataFrame.
            
        Returns:
            List of cluster names.
        """
        names = []
        
        for _, row in profiles_df.iterrows():
            avg_income = row.get('avg_income', 0)
            avg_spending = row.get('avg_spending', 0)
            size_pct = row.get('size_pct', 0)
            
            # Generate name based on characteristics
            if avg_income > 80 and avg_spending > 70:
                name = "High-Value Customers"
            elif avg_income > 60 and avg_spending > 50:
                name = "Premium Customers"
            elif avg_income < 40 and avg_spending < 30:
                name = "Budget-Conscious"
            elif avg_spending > avg_income * 0.8:
                name = "High Spenders"
            elif avg_income > 70 and avg_spending < 40:
                name = "High Income, Low Spending"
            else:
                name = f"Segment {int(row['cluster_id'])}"
            
            names.append(name)
        
        return names
    
    def calculate_segment_value(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate segment value metrics.
        
        Args:
            df: Customer DataFrame.
            labels: Cluster labels.
            
        Returns:
            Dictionary with segment value metrics.
        """
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        segment_metrics = {}
        
        for cluster_id in sorted(np.unique(labels)):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Calculate segment value
            if 'annual_income' in cluster_data.columns:
                total_income = cluster_data['annual_income'].sum()
                avg_income = cluster_data['annual_income'].mean()
            else:
                total_income = 0
                avg_income = 0
            
            if 'purchase_frequency' in cluster_data.columns and 'avg_order_value' in cluster_data.columns:
                estimated_revenue = (cluster_data['purchase_frequency'] * cluster_data['avg_order_value']).sum()
            else:
                estimated_revenue = 0
            
            segment_metrics[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'total_income': total_income,
                'avg_income': avg_income,
                'estimated_revenue': estimated_revenue,
                'revenue_per_customer': estimated_revenue / len(cluster_data) if len(cluster_data) > 0 else 0
            }
        
        return segment_metrics
    
    def calculate_customer_lifetime_value(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        discount_rate: float = 0.1,
        retention_rate: float = 0.8
    ) -> pd.DataFrame:
        """Calculate estimated customer lifetime value by segment.
        
        Args:
            df: Customer DataFrame.
            labels: Cluster labels.
            discount_rate: Discount rate for CLV calculation.
            retention_rate: Customer retention rate.
            
        Returns:
            DataFrame with CLV metrics by segment.
        """
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        clv_data = []
        
        for cluster_id in sorted(np.unique(labels)):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            # Calculate average annual value
            if 'purchase_frequency' in cluster_data.columns and 'avg_order_value' in cluster_data.columns:
                avg_annual_value = (cluster_data['purchase_frequency'] * cluster_data['avg_order_value']).mean()
            else:
                avg_annual_value = cluster_data.get('spending_score', 0).mean()
            
            # Calculate CLV using simplified formula
            # CLV = (Annual Value * Retention Rate) / (1 + Discount Rate - Retention Rate)
            clv = (avg_annual_value * retention_rate) / (1 + discount_rate - retention_rate)
            
            clv_data.append({
                'cluster_id': cluster_id,
                'avg_annual_value': avg_annual_value,
                'clv': clv,
                'size': len(cluster_data)
            })
        
        clv_df = pd.DataFrame(clv_data)
        clv_df['total_clv'] = clv_df['clv'] * clv_df['size']
        
        return clv_df.sort_values('clv', ascending=False)


class SegmentationInsights:
    """Generate insights and recommendations from segmentation results."""
    
    def __init__(self) -> None:
        """Initialize the insights generator."""
        pass
    
    def generate_recommendations(
        self, 
        profiles_df: pd.DataFrame,
        clv_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Generate business recommendations based on segmentation.
        
        Args:
            profiles_df: Cluster profiles DataFrame.
            clv_df: CLV DataFrame.
            
        Returns:
            Dictionary with recommendations by cluster.
        """
        recommendations = {}
        
        for _, profile in profiles_df.iterrows():
            cluster_id = int(profile['cluster_id'])
            cluster_name = profile['cluster_name']
            
            cluster_recs = []
            
            # High-value customer recommendations
            if 'High-Value' in cluster_name or 'Premium' in cluster_name:
                cluster_recs.extend([
                    "Implement VIP customer service program",
                    "Offer exclusive products and early access",
                    "Create personalized loyalty rewards",
                    "Develop premium customer retention campaigns"
                ])
            
            # Budget-conscious recommendations
            elif 'Budget' in cluster_name:
                cluster_recs.extend([
                    "Focus on value-oriented messaging",
                    "Offer bundle deals and discounts",
                    "Implement price-sensitive marketing",
                    "Create budget-friendly product lines"
                ])
            
            # High spenders recommendations
            elif 'High Spenders' in cluster_name:
                cluster_recs.extend([
                    "Increase product recommendations",
                    "Implement cross-selling strategies",
                    "Create impulse purchase opportunities",
                    "Develop premium product showcases"
                ])
            
            # High income, low spending recommendations
            elif 'High Income' in cluster_name and 'Low Spending' in cluster_name:
                cluster_recs.extend([
                    "Investigate barriers to purchase",
                    "Create premium product awareness campaigns",
                    "Implement lifestyle-based marketing",
                    "Develop trust-building initiatives"
                ])
            
            # General recommendations
            cluster_recs.extend([
                f"Monitor segment size trends (currently {profile['size_pct']:.1f}% of customers)",
                f"Track CLV performance (current: ${profile.get('clv', 0):.2f})",
                "Implement segment-specific communication strategies",
                "Regularly review and update segmentation model"
            ])
            
            recommendations[f"{cluster_name} (Cluster {cluster_id})"] = cluster_recs
        
        return recommendations
    
    def identify_opportunities(self, profiles_df: pd.DataFrame) -> List[str]:
        """Identify business opportunities from segmentation.
        
        Args:
            profiles_df: Cluster profiles DataFrame.
            
        Returns:
            List of identified opportunities.
        """
        opportunities = []
        
        # Analyze segment sizes
        large_segments = profiles_df[profiles_df['size_pct'] > 30]
        small_segments = profiles_df[profiles_df['size_pct'] < 10]
        
        if len(large_segments) > 0:
            opportunities.append(
                f"Large segments detected: {', '.join(large_segments['cluster_name'].tolist())} "
                f"- consider sub-segmentation for more targeted strategies"
            )
        
        if len(small_segments) > 0:
            opportunities.append(
                f"Small segments detected: {', '.join(small_segments['cluster_name'].tolist())} "
                f"- investigate growth potential or consider merging strategies"
            )
        
        # Analyze income-spending patterns
        high_income_low_spending = profiles_df[
            (profiles_df.get('avg_income', 0) > 70) & 
            (profiles_df.get('avg_spending', 0) < 40)
        ]
        
        if len(high_income_low_spending) > 0:
            opportunities.append(
                "High-income, low-spending segments identified - "
                "significant revenue opportunity with targeted marketing"
            )
        
        # Analyze spending patterns
        high_spenders = profiles_df[profiles_df.get('avg_spending', 0) > 70]
        if len(high_spenders) > 0:
            opportunities.append(
                "High-spending segments identified - "
                "focus on retention and expansion strategies"
            )
        
        return opportunities
