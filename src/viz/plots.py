"""Visualization module for customer segmentation analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SegmentationVisualizer:
    """Create visualizations for customer segmentation analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300) -> None:
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size.
            dpi: Figure DPI.
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_clusters_2d(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        feature_columns: List[str],
        title: str = "Customer Segmentation",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot 2D cluster visualization.
        
        Args:
            df: Customer DataFrame.
            labels: Cluster labels.
            feature_columns: Feature columns to plot.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot different feature combinations
        feature_pairs = [
            ('annual_income', 'spending_score'),
            ('age', 'spending_score'),
            ('annual_income', 'avg_order_value'),
            ('purchase_frequency', 'avg_order_value')
        ]
        
        for i, (x_feat, y_feat) in enumerate(feature_pairs):
            if i >= len(axes):
                break
                
            if x_feat in df.columns and y_feat in df.columns:
                ax = axes[i]
                
                # Create scatter plot
                scatter = ax.scatter(
                    df[x_feat], 
                    df[y_feat], 
                    c=labels, 
                    cmap='viridis', 
                    alpha=0.7,
                    s=50
                )
                
                ax.set_xlabel(x_feat.replace('_', ' ').title())
                ax.set_ylabel(y_feat.replace('_', ' ').title())
                ax.set_title(f'{x_feat.replace("_", " ").title()} vs {y_feat.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # Hide unused subplots
        for i in range(len(feature_pairs), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved cluster plot to {save_path}")
        
        return fig
    
    def plot_cluster_profiles(
        self, 
        profiles_df: pd.DataFrame,
        title: str = "Cluster Profiles",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot cluster profile comparison.
        
        Args:
            profiles_df: Cluster profiles DataFrame.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cluster sizes
        ax1 = axes[0, 0]
        bars = ax1.bar(profiles_df['cluster_name'], profiles_df['size'])
        ax1.set_title('Cluster Sizes')
        ax1.set_ylabel('Number of Customers')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Income vs Spending
        ax2 = axes[0, 1]
        if 'avg_income' in profiles_df.columns and 'avg_spending' in profiles_df.columns:
            scatter = ax2.scatter(
                profiles_df['avg_income'], 
                profiles_df['avg_spending'],
                s=profiles_df['size'] * 2,  # Size by cluster size
                alpha=0.7,
                c=range(len(profiles_df)),
                cmap='viridis'
            )
            ax2.set_xlabel('Average Income')
            ax2.set_ylabel('Average Spending Score')
            ax2.set_title('Income vs Spending by Cluster')
            ax2.grid(True, alpha=0.3)
            
            # Add cluster labels
            for i, row in profiles_df.iterrows():
                ax2.annotate(
                    row['cluster_name'], 
                    (row['avg_income'], row['avg_spending']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8
                )
        
        # Purchase frequency vs Order value
        ax3 = axes[1, 0]
        if 'avg_purchase_freq' in profiles_df.columns and 'avg_order_value' in profiles_df.columns:
            ax3.scatter(
                profiles_df['avg_purchase_freq'], 
                profiles_df['avg_order_value'],
                s=profiles_df['size'] * 2,
                alpha=0.7,
                c=range(len(profiles_df)),
                cmap='viridis'
            )
            ax3.set_xlabel('Average Purchase Frequency')
            ax3.set_ylabel('Average Order Value')
            ax3.set_title('Purchase Behavior by Cluster')
            ax3.grid(True, alpha=0.3)
        
        # Size distribution pie chart
        ax4 = axes[1, 1]
        ax4.pie(
            profiles_df['size'], 
            labels=profiles_df['cluster_name'],
            autopct='%1.1f%%',
            startangle=90
        )
        ax4.set_title('Cluster Size Distribution')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved profile plot to {save_path}")
        
        return fig
    
    def plot_clv_analysis(
        self, 
        clv_df: pd.DataFrame,
        title: str = "Customer Lifetime Value Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot CLV analysis.
        
        Args:
            clv_df: CLV DataFrame.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CLV by cluster
        ax1 = axes[0]
        bars = ax1.bar(
            range(len(clv_df)), 
            clv_df['clv'],
            color='skyblue',
            alpha=0.7
        )
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Customer Lifetime Value ($)')
        ax1.set_title('CLV by Cluster')
        ax1.set_xticks(range(len(clv_df)))
        ax1.set_xticklabels([f'Cluster {i}' for i in clv_df['cluster_id']], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}', ha='center', va='bottom')
        
        # Total CLV contribution
        ax2 = axes[1]
        ax2.pie(
            clv_df['total_clv'], 
            labels=[f'Cluster {i}' for i in clv_df['cluster_id']],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Total CLV Contribution by Cluster')
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved CLV plot to {save_path}")
        
        return fig
    
    def create_interactive_dashboard(
        self, 
        df: pd.DataFrame, 
        labels: np.ndarray,
        profiles_df: pd.DataFrame,
        clv_df: pd.DataFrame
    ) -> go.Figure:
        """Create interactive Plotly dashboard.
        
        Args:
            df: Customer DataFrame.
            labels: Cluster labels.
            profiles_df: Cluster profiles DataFrame.
            clv_df: CLV DataFrame.
            
        Returns:
            Plotly figure.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Income vs Spending Score',
                'Age vs Purchase Frequency', 
                'Cluster Sizes',
                'Customer Lifetime Value'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Income vs Spending scatter
        fig.add_trace(
            go.Scatter(
                x=df['annual_income'],
                y=df['spending_score'],
                mode='markers',
                marker=dict(
                    color=labels,
                    colorscale='viridis',
                    size=8,
                    opacity=0.7
                ),
                text=[f'Customer {i}' for i in df['customer_id']],
                hovertemplate='<b>%{text}</b><br>' +
                             'Income: $%{x}k<br>' +
                             'Spending Score: %{y}<br>' +
                             'Cluster: %{marker.color}<extra></extra>',
                name='Customers'
            ),
            row=1, col=1
        )
        
        # Age vs Purchase Frequency scatter
        if 'purchase_frequency' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['age'],
                    y=df['purchase_frequency'],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='viridis',
                        size=8,
                        opacity=0.7
                    ),
                    text=[f'Customer {i}' for i in df['customer_id']],
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Age: %{x}<br>' +
                                 'Purchase Frequency: %{y}<br>' +
                                 'Cluster: %{marker.color}<extra></extra>',
                    name='Customers',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Cluster sizes bar chart
        fig.add_trace(
            go.Bar(
                x=profiles_df['cluster_name'],
                y=profiles_df['size'],
                marker_color='lightblue',
                text=profiles_df['size'],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Size: %{y}<br>' +
                             'Percentage: %{customdata:.1f}%<extra></extra>',
                customdata=profiles_df['size_pct'],
                name='Cluster Size'
            ),
            row=2, col=1
        )
        
        # CLV bar chart
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in clv_df['cluster_id']],
                y=clv_df['clv'],
                marker_color='lightgreen',
                text=[f'${val:.0f}' for val in clv_df['clv']],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'CLV: $%{y:.0f}<br>' +
                             'Size: %{customdata}<extra></extra>',
                customdata=clv_df['size'],
                name='CLV'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Customer Segmentation Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=1)
        
        if 'purchase_frequency' in df.columns:
            fig.update_xaxes(title_text="Age", row=1, col=2)
            fig.update_yaxes(title_text="Purchase Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Cluster", row=2, col=1)
        fig.update_yaxes(title_text="Number of Customers", row=2, col=1)
        
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Customer Lifetime Value ($)", row=2, col=2)
        
        return fig
    
    def plot_elbow_method(
        self, 
        k_range: List[int], 
        inertias: List[float],
        silhouette_scores: List[float],
        title: str = "Elbow Method and Silhouette Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot elbow method and silhouette analysis.
        
        Args:
            k_range: Range of k values tested.
            inertias: Inertia values for each k.
            silhouette_scores: Silhouette scores for each k.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow method
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette analysis
        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Highlight optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
        ax2.text(optimal_k, max(silhouette_scores) * 0.9, 
                f'Optimal k = {optimal_k}', 
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved elbow plot to {save_path}")
        
        return fig
