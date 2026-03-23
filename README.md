# Customer Segmentation Analysis

A comprehensive customer segmentation analysis tool that uses multiple clustering algorithms to identify distinct customer groups and provide actionable business insights.

## Overview

This project provides a complete customer segmentation solution with:
- Multiple clustering algorithms (K-Means, Gaussian Mixture Models, HDBSCAN)
- Comprehensive evaluation metrics
- Business-focused insights and recommendations
- Interactive visualizations and dashboards
- Production-ready code structure

## Features

### Core Capabilities
- **Data Generation**: Synthetic customer data with realistic characteristics
- **Multiple Algorithms**: K-Means, GMM, and HDBSCAN clustering
- **Automatic Optimization**: Elbow method for optimal cluster selection
- **Comprehensive Evaluation**: Silhouette score, Calinski-Harabasz, Davies-Bouldin metrics
- **Business Metrics**: Customer Lifetime Value (CLV), segment profiles, revenue analysis

### Advanced Features
- **Interactive Demo**: Streamlit-based web application
- **Visualization Suite**: Static and interactive plots with Plotly
- **Business Insights**: Automated recommendations and opportunity identification
- **Reproducible Results**: Deterministic seeding and configuration management
- **Production Ready**: Type hints, logging, error handling, and documentation

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Customer-Segmentation-Analysis.git
cd Customer-Segmentation-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Run the complete analysis**:
```bash
python scripts/run_analysis.py
```

2. **Launch the interactive demo**:
```bash
streamlit run demo/app.py
```

3. **Use the original sample data**:
```python
from scripts.run_analysis import CustomerSegmentationAnalysis

analysis = CustomerSegmentationAnalysis()
results = analysis.run_complete_analysis(use_sample=True)
analysis.print_summary()
```

## Project Structure

```
├── src/                          # Source code
│   ├── data/                     # Data generation and preprocessing
│   │   └── generator.py         # Synthetic data generation
│   ├── models/                   # Clustering models
│   │   └── segmentation.py      # K-Means, GMM, HDBSCAN implementations
│   ├── eval/                     # Evaluation and business metrics
│   │   └── business_metrics.py  # CLV, profiles, insights
│   ├── viz/                      # Visualization
│   │   └── plots.py            # Static and interactive plots
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   └── config.yaml             # Main configuration
├── scripts/                      # Analysis scripts
│   └── run_analysis.py         # Main analysis pipeline
├── demo/                         # Interactive demo
│   └── app.py                  # Streamlit application
├── tests/                        # Unit tests
├── assets/                       # Generated outputs
├── notebooks/                     # Jupyter notebooks
└── requirements.txt              # Dependencies
```

## Dataset Schema

### Synthetic Customer Data
The tool generates realistic customer data with the following features:

- **age**: Customer age (18-80 years)
- **annual_income**: Annual income in thousands (20-150k)
- **spending_score**: Spending behavior score (1-100)
- **purchase_frequency**: Number of purchases per year (1-50)
- **avg_order_value**: Average order value in dollars (10-500)

### Sample Data
For backward compatibility, the original sample dataset is available:
- **Age**: Customer age
- **Annual_Income_k$**: Annual income in thousands
- **Spending_Score**: Spending behavior score

## Configuration

The analysis can be customized through `configs/config.yaml`:

```yaml
data:
  synthetic:
    n_customers: 1000
    random_seed: 42
    noise_level: 0.1

models:
  kmeans:
    n_clusters: 5
    random_state: 42
  gmm:
    n_components: 5
    random_state: 42
  hdbscan:
    min_cluster_size: 20
    min_samples: 10

evaluation:
  metrics:
    - silhouette_score
    - calinski_harabasz_score
    - davies_bouldin_score
```

## Model Performance

### Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Score**: Average similarity ratio of clusters
- **Adjusted Rand Index**: External validation (when true labels available)

### Business Metrics
- **Customer Lifetime Value (CLV)**: Estimated customer value by segment
- **Segment Profiles**: Demographic and behavioral characteristics
- **Revenue Analysis**: Income and spending patterns
- **Size Distribution**: Customer distribution across segments

## Interactive Demo

The Streamlit demo provides:
- **Data Generation**: Customizable parameters
- **Model Training**: Multiple algorithms with performance comparison
- **Visualization**: Interactive plots and cluster analysis
- **Business Insights**: Automated recommendations and opportunities
- **Export Capabilities**: Save results and visualizations

Access the demo at: `http://localhost:8501`

## Business Applications

### Customer Segmentation Use Cases
- **Marketing Strategy**: Targeted campaigns based on customer profiles
- **Product Development**: Feature prioritization by segment needs
- **Pricing Strategy**: Segment-specific pricing models
- **Customer Retention**: Identify at-risk customer groups
- **Cross-selling**: Product recommendations by segment behavior

### Typical Customer Segments
- **High-Value Customers**: High income, high spending
- **Premium Customers**: Moderate income, high spending
- **Budget-Conscious**: Low income, low spending
- **High Spenders**: High spending relative to income
- **High Income, Low Spending**: Potential growth opportunity

## API Reference

### Core Classes

#### `CustomerSegmentationAnalysis`
Main analysis class that orchestrates the complete pipeline.

```python
analysis = CustomerSegmentationAnalysis(config_path="configs/config.yaml")
results = analysis.run_complete_analysis()
```

#### `CustomerDataGenerator`
Generates synthetic customer data with realistic characteristics.

```python
generator = CustomerDataGenerator(random_seed=42)
df = generator.generate_customer_data(n_customers=1000)
```

#### `SegmentationEvaluator`
Evaluates clustering models using multiple metrics.

```python
evaluator = SegmentationEvaluator()
metrics = evaluator.evaluate_model(model, X, true_labels)
```

## Development

### Setup Development Environment
```bash
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Type Checking
```bash
mypy src/ scripts/ demo/
```

## Limitations and Considerations

### Model Limitations
- **K-Means**: Assumes spherical clusters, sensitive to initialization
- **GMM**: Assumes Gaussian distributions, may struggle with complex shapes
- **HDBSCAN**: May identify noise points, sensitive to parameters

### Data Considerations
- Synthetic data may not reflect real-world complexity
- Feature scaling is critical for clustering performance
- Optimal number of clusters depends on business context

### Business Context
- Segmentation should be validated with domain experts
- Regular model updates may be necessary as customer behavior changes
- Consider external factors (seasonality, market changes)

## Privacy and Compliance

### Data Privacy
- All generated data is synthetic and contains no real customer information
- No PII (Personally Identifiable Information) is processed or stored
- Data generation uses anonymized statistical patterns

### Compliance Notes
- This tool is designed for research and educational purposes
- Results should not be used for automated decision-making without human review
- Consider applicable data protection regulations in your jurisdiction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create an issue in the repository
- Review the documentation and examples
- Check the demo application for usage patterns

## Disclaimer

**IMPORTANT**: This tool is designed for research and educational purposes. The analysis and recommendations provided are experimental and should not be used for automated business decisions without human review. Always validate results with domain experts and consider business context before implementing any strategies.

The synthetic data generated by this tool does not represent real customers and should not be used to make decisions about actual customer segments without proper validation and domain expertise.
# Customer-Segmentation-Analysis
