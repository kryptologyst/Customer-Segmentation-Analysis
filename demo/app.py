"""Streamlit demo for customer segmentation analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import CustomerDataGenerator, DataPreprocessor
from models.segmentation import KMeansSegmentation, GaussianMixtureSegmentation, HDBSCANSegmentation
from eval.business_metrics import BusinessMetricsCalculator, SegmentationInsights
from viz.plots import SegmentationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Customer Segmentation Analysis</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This is an experimental research and educational tool.</strong> 
    The analysis and recommendations provided are for demonstration purposes only 
    and should not be used for automated business decisions without human review. 
    Always validate results with domain experts and consider business context 
    before implementing any strategies.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Data generation parameters
st.sidebar.subheader("Data Parameters")
n_customers = st.sidebar.slider("Number of Customers", 100, 2000, 1000)
random_seed = st.sidebar.number_input("Random Seed", 1, 1000, 42)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, 0.01)

# Model parameters
st.sidebar.subheader("Model Parameters")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)
min_cluster_size = st.sidebar.slider("Min Cluster Size (HDBSCAN)", 5, 50, 20)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🤖 Model Training", 
    "📈 Results", 
    "💼 Business Insights", 
    "📋 Recommendations"
])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate New Data", type="primary"):
            with st.spinner("Generating customer data..."):
                # Generate data
                generator = CustomerDataGenerator(random_seed=random_seed)
                df = generator.generate_customer_data(
                    n_customers=n_customers,
                    noise_level=noise_level
                )
                
                # Store in session state
                st.session_state.df = df
                st.session_state.data_generated = True
                st.session_state.models_trained = False
                
                st.success(f"Generated {len(df)} customers successfully!")
    
    with col2:
        if st.session_state.data_generated:
            st.metric("Total Customers", len(st.session_state.df))
            st.metric("Features", len(st.session_state.df.columns) - 2)  # Exclude customer_id and true_segment
    
    if st.session_state.data_generated:
        df = st.session_state.df
        
        # Data summary
        st.subheader("Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Age", f"{df['age'].mean():.1f}")
        with col2:
            st.metric("Avg Income", f"${df['annual_income'].mean():.0f}k")
        with col3:
            st.metric("Avg Spending", f"{df['spending_score'].mean():.1f}")
        with col4:
            st.metric("Avg Order Value", f"${df['avg_order_value'].mean():.0f}")
        
        # Data visualization
        st.subheader("Data Distribution")
        
        # Create scatter plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income vs Spending', 'Age vs Purchase Frequency', 
                          'Income Distribution', 'Spending Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Income vs Spending
        fig.add_trace(
            go.Scatter(
                x=df['annual_income'],
                y=df['spending_score'],
                mode='markers',
                marker=dict(size=6, opacity=0.6),
                name='Customers'
            ),
            row=1, col=1
        )
        
        # Age vs Purchase Frequency
        fig.add_trace(
            go.Scatter(
                x=df['age'],
                y=df['purchase_frequency'],
                mode='markers',
                marker=dict(size=6, opacity=0.6),
                name='Customers',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Income distribution
        fig.add_trace(
            go.Histogram(x=df['annual_income'], name='Income', showlegend=False),
            row=2, col=1
        )
        
        # Spending distribution
        fig.add_trace(
            go.Histogram(x=df['spending_score'], name='Spending', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
        fig.update_yaxes(title_text="Spending Score", row=1, col=1)
        fig.update_xaxes(title_text="Age", row=1, col=2)
        fig.update_yaxes(title_text="Purchase Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Annual Income (k$)", row=2, col=1)
        fig.update_xaxes(title_text="Spending Score", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.header("Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Train Models", type="primary"):
                with st.spinner("Training segmentation models..."):
                    df = st.session_state.df
                    
                    # Preprocess data
                    feature_columns = ['age', 'annual_income', 'spending_score', 
                                     'purchase_frequency', 'avg_order_value']
                    preprocessor = DataPreprocessor()
                    X_scaled = preprocessor.fit_transform(df, feature_columns)
                    
                    # Train models
                    models = {}
                    results = {}
                    
                    # K-Means
                    kmeans = KMeansSegmentation(n_clusters=n_clusters, random_state=random_seed)
                    kmeans.fit(X_scaled)
                    models['K-Means'] = kmeans
                    
                    # GMM
                    gmm = GaussianMixtureSegmentation(n_components=n_clusters, random_state=random_seed)
                    gmm.fit(X_scaled)
                    models['GMM'] = gmm
                    
                    # HDBSCAN
                    hdbscan_model = HDBSCANSegmentation(min_cluster_size=min_cluster_size)
                    hdbscan_model.fit(X_scaled)
                    models['HDBSCAN'] = hdbscan_model
                    
                    # Store results
                    st.session_state.models = models
                    st.session_state.X_scaled = X_scaled
                    st.session_state.feature_columns = feature_columns
                    st.session_state.models_trained = True
                    
                    st.success("Models trained successfully!")
        
        with col2:
            if st.session_state.models_trained:
                st.metric("Models Trained", len(st.session_state.models))
                st.metric("Features Used", len(st.session_state.feature_columns))
        
        if st.session_state.models_trained:
            st.subheader("Model Performance")
            
            # Calculate metrics
            models = st.session_state.models
            X_scaled = st.session_state.X_scaled
            
            metrics_data = []
            for name, model in models.items():
                labels = model.labels_
                
                # Skip noise points for HDBSCAN
                if name == 'HDBSCAN':
                    mask = labels != -1
                    if np.sum(mask) == 0:
                        continue
                    X_eval = X_scaled[mask]
                    labels_eval = labels[mask]
                else:
                    X_eval = X_scaled
                    labels_eval = labels
                
                if len(np.unique(labels_eval)) > 1:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(X_eval, labels_eval)
                    n_clusters = len(np.unique(labels_eval))
                    n_noise = np.sum(labels == -1) if name == 'HDBSCAN' else 0
                    
                    metrics_data.append({
                        'Model': name,
                        'Silhouette Score': silhouette,
                        'Clusters': n_clusters,
                        'Noise Points': n_noise
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Performance chart
                fig = px.bar(
                    metrics_df, 
                    x='Model', 
                    y='Silhouette Score',
                    title='Model Performance Comparison',
                    color='Silhouette Score',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Segmentation Results")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
    else:
        # Model selection
        model_name = st.selectbox(
            "Select Model to Visualize",
            list(st.session_state.models.keys())
        )
        
        if model_name:
            model = st.session_state.models[model_name]
            df = st.session_state.df
            X_scaled = st.session_state.X_scaled
            feature_columns = st.session_state.feature_columns
            
            labels = model.labels_
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Income vs Spending', 'Age vs Purchase Frequency', 
                              'Cluster Sizes', 'Feature Distribution'),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "box"}]]
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
                    name='Customers'
                ),
                row=1, col=1
            )
            
            # Age vs Purchase Frequency scatter
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
                    name='Customers',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            if -1 in unique_labels:  # Remove noise points
                mask = unique_labels != -1
                unique_labels = unique_labels[mask]
                counts = counts[mask]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {i}' for i in unique_labels],
                    y=counts,
                    name='Cluster Size',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Feature distribution by cluster
            for i, feature in enumerate(['annual_income', 'spending_score']):
                if feature in df.columns:
                    fig.add_trace(
                        go.Box(
                            y=df[feature],
                            x=[f'Cluster {label}' for label in labels],
                            name=feature,
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                    break
            
            fig.update_layout(height=800, showlegend=True)
            fig.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
            fig.update_yaxes(title_text="Spending Score", row=1, col=1)
            fig.update_xaxes(title_text="Age", row=1, col=2)
            fig.update_yaxes(title_text="Purchase Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Cluster", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            cluster_stats = []
            for cluster_id in sorted(np.unique(labels)):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_data = df[labels == cluster_id]
                stats = {
                    'Cluster': cluster_id,
                    'Size': len(cluster_data),
                    'Avg Age': cluster_data['age'].mean(),
                    'Avg Income': cluster_data['annual_income'].mean(),
                    'Avg Spending': cluster_data['spending_score'].mean(),
                    'Avg Order Value': cluster_data['avg_order_value'].mean()
                }
                cluster_stats.append(stats)
            
            if cluster_stats:
                stats_df = pd.DataFrame(cluster_stats)
                st.dataframe(stats_df, use_container_width=True)

with tab4:
    st.header("Business Insights")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
    else:
        # Use K-Means for business analysis
        kmeans_model = st.session_state.models['K-Means']
        df = st.session_state.df
        labels = kmeans_model.labels_
        
        # Calculate business metrics
        business_calc = BusinessMetricsCalculator()
        feature_columns = st.session_state.feature_columns
        
        profiles_df = business_calc.calculate_cluster_profiles(df, labels, feature_columns)
        clv_df = business_calc.calculate_customer_lifetime_value(df, labels)
        
        # Display cluster profiles
        st.subheader("Cluster Profiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster sizes
            fig = px.pie(
                profiles_df, 
                values='size', 
                names='cluster_name',
                title='Customer Distribution by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # CLV analysis
            fig = px.bar(
                clv_df, 
                x='cluster_id', 
                y='clv',
                title='Customer Lifetime Value by Segment',
                labels={'cluster_id': 'Cluster', 'clv': 'CLV ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed profiles table
        st.subheader("Detailed Segment Profiles")
        
        # Format the profiles table for display
        display_profiles = profiles_df.copy()
        display_profiles = display_profiles.round(2)
        
        st.dataframe(display_profiles, use_container_width=True)
        
        # CLV table
        st.subheader("Customer Lifetime Value Analysis")
        st.dataframe(clv_df.round(2), use_container_width=True)

with tab5:
    st.header("Business Recommendations")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab.")
    else:
        # Generate insights
        kmeans_model = st.session_state.models['K-Means']
        df = st.session_state.df
        labels = kmeans_model.labels_
        
        business_calc = BusinessMetricsCalculator()
        insights_gen = SegmentationInsights()
        feature_columns = st.session_state.feature_columns
        
        profiles_df = business_calc.calculate_cluster_profiles(df, labels, feature_columns)
        clv_df = business_calc.calculate_customer_lifetime_value(df, labels)
        
        recommendations = insights_gen.generate_recommendations(profiles_df, clv_df)
        opportunities = insights_gen.identify_opportunities(profiles_df)
        
        # Display recommendations
        st.subheader("Segment-Specific Recommendations")
        
        for segment, recs in recommendations.items():
            with st.expander(f"📋 {segment}"):
                for i, rec in enumerate(recs, 1):
                    st.write(f"{i}. {rec}")
        
        # Display opportunities
        st.subheader("Key Business Opportunities")
        
        for opportunity in opportunities:
            st.info(f"💡 {opportunity}")
        
        # Action items
        st.subheader("Recommended Next Steps")
        
        action_items = [
            "Validate segmentation results with business stakeholders",
            "Develop segment-specific marketing campaigns",
            "Implement customer journey mapping for each segment",
            "Set up monitoring dashboards for segment performance",
            "Plan A/B tests for segment-specific strategies",
            "Regularly review and update segmentation model"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.write(f"{i}. {item}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Customer Segmentation Analysis Demo | Experimental Research Tool</p>
    <p><strong>Disclaimer:</strong> This tool is for educational purposes only. 
    Do not use for automated business decisions without human review.</p>
</div>
""", unsafe_allow_html=True)
