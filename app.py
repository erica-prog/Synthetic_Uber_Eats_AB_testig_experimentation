import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import skew, kurtosis, shapiro, mannwhitneyu, ttest_ind
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Uber Eats A/B Test Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Uber Eats theme - White background with green/black accents
st.markdown("""
<style>
    /* Main app styling - White background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 2px solid #5fb709;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Headers styling - Black text */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Metric cards with dark green */
    .metric-card {
        background: linear-gradient(135deg, #2d5016 0%, #1f3810 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 1px solid #5fb709;
    }
    
    /* Success cards with bright green */
    .success-card {
        background: linear-gradient(135deg, #5fb709 0%, #4a9207 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 1px solid #5fb709;
    }
    
    /* Warning cards with red accent */
    .warning-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 1px solid #e74c3c;
    }
    
    /* Executive summary with dark green */
    .executive-summary {
        background: linear-gradient(135deg, #2d5016 0%, #1f3810 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 2px solid #5fb709;
    }
    
    /* Streamlit metrics styling */
    .css-1xarl3l {
        background-color: #f8f9fa;
        border: 2px solid #5fb709;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Text color adjustments */
    .css-10trblm {
        color: #000000;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #5fb709;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #5fb709;
        color: #000000;
    }
    
    /* Checkbox styling */
    .stCheckbox > div {
        color: #000000;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #5fb709;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #4a9207;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #5fb709;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        color: #000000;
        border: 2px solid #5fb709;
    }
    
    /* Custom Uber Eats title styling */
    .uber-title {
        background: linear-gradient(90deg, #5fb709 0%, #000000 50%, #5fb709 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Custom section headers */
    .section-header {
        background-color: #5fb709;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    /* Sidebar text color */
    .css-1d391kg .stMarkdown {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Configuration class
class ExperimentConfig:
    EXPERIMENT_NAME = "smart_recommendations_v1"
    START_DATE = "2024-10-01"
    END_DATE = "2024-10-15"
    SIGNIFICANCE_LEVEL = 0.05
    STATISTICAL_POWER = 0.8
    
    MIN_PRACTICAL_SIGNIFICANCE = {
        'satisfaction_score': 0.1,
        'order_value_usd': 1.0,
        'order_completed': 0.02,
        'time_to_order_minutes': 0.5
    }

config = ExperimentConfig()

# Initialize session state for data
if 'data_initialized' not in st.session_state:
    # Generate synthetic data matching the experiment results
    np.random.seed(42)
    
    n_control = 199221
    n_treatment = 199760
    
    # Control group data
    control_satisfaction = np.random.normal(3.309, 0.229, n_control)
    control_completion = np.random.binomial(1, 0.879, n_control)
    control_order_value = np.random.normal(28.514, 4.495, n_control)
    control_time_to_order = np.random.normal(8.502, 0.999, n_control)
    control_items_viewed = np.random.normal(11.003, 2.825, n_control)
    control_session_duration = np.random.normal(420.444, 99.893, n_control)
    
    # Treatment group data
    treatment_satisfaction = np.random.normal(3.710, 0.189, n_treatment)
    treatment_completion = np.random.binomial(1, 0.930, n_treatment)
    treatment_order_value = np.random.normal(31.685, 4.988, n_treatment)
    treatment_time_to_order = np.random.normal(7.298, 0.800, n_treatment)
    treatment_items_viewed = np.random.normal(16.996, 3.467, n_treatment)
    treatment_session_duration = np.random.normal(480.301, 120.259, n_treatment)
    
    # Create DataFrames
    control_df = pd.DataFrame({
        'group': 'control',
        'satisfaction_score': control_satisfaction,
        'order_completed': control_completion,
        'order_value_usd': control_order_value,
        'time_to_order_minutes': control_time_to_order,
        'items_viewed': control_items_viewed,
        'app_session_duration_seconds': control_session_duration,
        'user_id': range(n_control)
    })
    
    treatment_df = pd.DataFrame({
        'group': 'treatment',
        'satisfaction_score': treatment_satisfaction,
        'order_completed': treatment_completion,
        'order_value_usd': treatment_order_value,
        'time_to_order_minutes': treatment_time_to_order,
        'items_viewed': treatment_items_viewed,
        'app_session_duration_seconds': treatment_session_duration,
        'user_id': range(n_control, n_control + n_treatment)
    })
    
    # Add recommendation data for treatment group
    treatment_df['recommendations_shown'] = np.random.normal(9.0, 2.45, n_treatment)
    treatment_df['recommendations_clicked'] = np.random.binomial(1, 0.967, n_treatment) * np.random.poisson(3.15, n_treatment)
    treatment_df['ordered_from_recommendation'] = np.random.binomial(1, 0.280, n_treatment)
    
    # Add zeros for control group
    control_df['recommendations_shown'] = 0
    control_df['recommendations_clicked'] = 0
    control_df['ordered_from_recommendation'] = 0
    
    # Add segmentation variables
    cities = ['Los Angeles', 'San Diego', 'Dallas', 'New York', 'Philadelphia', 
              'San Antonio', 'Houston', 'San Jose', 'Chicago', 'Phoenix']
    age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
    frequency_segments = ['Low', 'Medium', 'High']
    
    control_df['city'] = np.random.choice(cities, n_control)
    control_df['age_group'] = np.random.choice(age_groups, n_control)
    control_df['user_frequency_segment'] = np.random.choice(frequency_segments, n_control)
    
    treatment_df['city'] = np.random.choice(cities, n_treatment)
    treatment_df['age_group'] = np.random.choice(age_groups, n_treatment)
    treatment_df['user_frequency_segment'] = np.random.choice(frequency_segments, n_treatment)
    
    # Add temporal data
    base_date = pd.Timestamp('2024-10-01')
    control_df['order_timestamp'] = base_date + pd.to_timedelta(np.random.randint(0, 14*24*60, n_control), unit='m')
    treatment_df['order_timestamp'] = base_date + pd.to_timedelta(np.random.randint(0, 14*24*60, n_treatment), unit='m')
    
    # Combine data
    st.session_state.df = pd.concat([control_df, treatment_df], ignore_index=True)
    st.session_state.data_initialized = True

# Load data
df = st.session_state.df

# Title and header
st.title("🍔 Uber Eats A/B Test Analysis Dashboard")
st.markdown("### Smart Recommendations Feature Analysis")

# Sidebar controls
st.sidebar.header("📊 Analysis Controls")
selected_metric = st.sidebar.selectbox(
    "Select Primary Metric to Analyze:",
    ["satisfaction_score", "order_completed", "order_value_usd"],
    format_func=lambda x: {
        "satisfaction_score": "Satisfaction Score",
        "order_completed": "Order Completion Rate", 
        "order_value_usd": "Average Order Value (USD)"
    }[x]
)

analysis_type = st.sidebar.radio(
    "Analysis Type:",
    ["Overview", "Statistical Testing", "Practical Significance", "Effect Size Analysis", 
     "Sequential Testing", "Business Impact", "Segmentation", "Data Quality"]
)

# Advanced statistical functions
def welch_ttest_with_ci(control_data, treatment_data):
    """Perform Welch's t-test with confidence intervals"""
    control_clean = control_data[~np.isnan(control_data)]
    treatment_clean = treatment_data[~np.isnan(treatment_data)]
    
    # Perform test
    stat, p_value = ttest_ind(treatment_clean, control_clean, equal_var=False)
    
    # Calculate statistics
    control_mean = np.mean(control_clean)
    treatment_mean = np.mean(treatment_clean)
    effect_size = treatment_mean - control_mean
    
    # Confidence interval using Welch-Satterthwaite
    n1, n2 = len(control_clean), len(treatment_clean)
    s1, s2 = np.std(control_clean, ddof=1), np.std(treatment_clean, ddof=1)
    
    se_diff = np.sqrt((s1**2/n1) + (s2**2/n2))
    df = ((s1**2/n1) + (s2**2/n2))**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    t_critical = stats.t.ppf(0.975, df)
    
    ci_lower = effect_size - t_critical * se_diff
    ci_upper = effect_size + t_critical * se_diff
    
    # Cohen's d
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    cohens_d = effect_size / pooled_std if pooled_std > 0 else 0
    
    return {
        'test_type': "Welch's t-test",
        'statistic': stat,
        'p_value': p_value,
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'effect_size': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'sample_size_control': n1,
        'sample_size_treatment': n2
    }

def mann_whitney_test_with_ci(control_data, treatment_data):
    """Perform Mann-Whitney U test with bootstrap CI"""
    control_clean = control_data[~np.isnan(control_data)]
    treatment_clean = treatment_data[~np.isnan(treatment_data)]
    
    stat, p_value = mannwhitneyu(treatment_clean, control_clean, alternative='two-sided')
    
    control_median = np.median(control_clean)
    treatment_median = np.median(treatment_clean)
    effect_size = treatment_median - control_median
    
    # Bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        control_sample = np.random.choice(control_clean, size=len(control_clean), replace=True)
        treatment_sample = np.random.choice(treatment_clean, size=len(treatment_clean), replace=True)
        diff = np.median(treatment_sample) - np.median(control_sample)
        bootstrap_diffs.append(diff)
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # Effect size approximation
    n1, n2 = len(control_clean), len(treatment_clean)
    cohens_d = 2 * (stat / (n1 * n2) - 0.5)
    
    return {
        'test_type': 'Mann-Whitney U',
        'statistic': stat,
        'p_value': p_value,
        'control_mean': control_median,
        'treatment_mean': treatment_median,
        'effect_size': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'sample_size_control': n1,
        'sample_size_treatment': n2
    }

def proportion_test_with_ci(control_successes, control_total, treatment_successes, treatment_total):
    """Proportion test with confidence intervals"""
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    effect_size = treatment_rate - control_rate
    
    # Combined proportion
    p_combined = (control_successes + treatment_successes) / (control_total + treatment_total)
    se = np.sqrt(p_combined * (1 - p_combined) * (1/control_total + 1/treatment_total))
    z_stat = effect_size / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Confidence interval
    se_diff = np.sqrt((control_rate * (1 - control_rate) / control_total) + 
                     (treatment_rate * (1 - treatment_rate) / treatment_total))
    ci_lower = effect_size - 1.96 * se_diff
    ci_upper = effect_size + 1.96 * se_diff
    
    # Cohen's h
    cohens_h = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate)))
    
    return {
        'test_type': 'Proportion Z-test',
        'statistic': z_stat,
        'p_value': p_value,
        'control_mean': control_rate,
        'treatment_mean': treatment_rate,
        'effect_size': effect_size,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_h,
        'sample_size_control': control_total,
        'sample_size_treatment': treatment_total
    }

def calculate_statistical_power(effect_size, n1, n2, alpha=0.05):
    """Calculate statistical power"""
    try:
        from scipy.stats import nct
        df = n1 + n2 - 2
        nc = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
        t_critical = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(t_critical, df, nc) + stats.nct.cdf(-t_critical, df, nc)
        return min(power, 1.0)
    except:
        return 0.8  # Default assumption

# Main analysis
control_data = df[df['group'] == 'control'][selected_metric].values
treatment_data = df[df['group'] == 'treatment'][selected_metric].values

# Determine appropriate test
if selected_metric == 'order_completed':
    control_successes = control_data.sum()
    control_total = len(control_data)
    treatment_successes = treatment_data.sum()
    treatment_total = len(treatment_data)
    
    test_results = proportion_test_with_ci(control_successes, control_total, 
                                         treatment_successes, treatment_total)
else:
    # Check skewness
    combined_data = np.concatenate([control_data, treatment_data])
    skewness = abs(skew(combined_data[~np.isnan(combined_data)]))
    
    if skewness > 2:
        test_results = mann_whitney_test_with_ci(control_data, treatment_data)
    else:
        test_results = welch_ttest_with_ci(control_data, treatment_data)

# Calculate additional metrics
relative_lift = (test_results['effect_size'] / test_results['control_mean']) * 100 if test_results['control_mean'] != 0 else 0
power = calculate_statistical_power(test_results['cohens_d'], 
                                   test_results['sample_size_control'], 
                                   test_results['sample_size_treatment'])

# Display results based on analysis type
if analysis_type == "Overview":
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Control Mean</h4>
            <h2>{test_results['control_mean']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Treatment Mean</h4>
            <h2>{test_results['treatment_mean']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="success-card">
            <h4>Effect Size</h4>
            <h2>{test_results['effect_size']:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="success-card">
            <h4>Relative Lift</h4>
            <h2>{relative_lift:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Comprehensive comparison chart
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("Distribution Comparison", "Box Plot Analysis",
                                     "Statistical Summary", "Effect Size Benchmark"))
    
    # Distribution overlay
    fig.add_trace(go.Histogram(x=control_data, name="Control", opacity=0.7, 
                              nbinsx=50, histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=treatment_data, name="Treatment", opacity=0.7, 
                              nbinsx=50, histnorm='probability density'), row=1, col=1)
    
    # Box plots
    fig.add_trace(go.Box(y=control_data, name="Control", boxpoints="outliers"), row=1, col=2)
    fig.add_trace(go.Box(y=treatment_data, name="Treatment", boxpoints="outliers"), row=1, col=2)
    
    # Effect size benchmarks
    effect_labels = ['Observed', 'Small (0.2)', 'Medium (0.5)', 'Large (0.8)']
    effect_values = [abs(test_results['cohens_d']), 0.2, 0.5, 0.8]
    colors = ['darkgreen', 'lightblue', 'orange', 'lightcoral']
    
    fig.add_trace(go.Bar(x=effect_labels, y=effect_values, marker_color=colors), row=2, col=2)
    
    # Add text annotation for statistical summary in empty subplot
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<b>Statistical Summary</b><br>" +
             f"Control: Mean={np.mean(control_data):.3f}, Std={np.std(control_data):.3f}<br>" +
             f"Treatment: Mean={np.mean(treatment_data):.3f}, Std={np.std(treatment_data):.3f}<br>" +
             f"Skewness: {skew(control_data):.3f} vs {skew(treatment_data):.3f}<br>" +
             f"Kurtosis: {kurtosis(control_data):.3f} vs {kurtosis(treatment_data):.3f}",
        xref="x3", yref="y3",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Comprehensive Analysis: {selected_metric}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Separate statistical summary table
    st.subheader("Statistical Summary")
    summary_data = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Median', 'Skewness', 'Kurtosis'],
        'Control': [np.mean(control_data), np.std(control_data), np.median(control_data),
                   skew(control_data), kurtosis(control_data)],
        'Treatment': [np.mean(treatment_data), np.std(treatment_data), np.median(treatment_data),
                     skew(treatment_data), kurtosis(treatment_data)]
    })
    st.dataframe(summary_data, use_container_width=True)

elif analysis_type == "Statistical Testing":
    st.subheader("🔬 Advanced Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Test Results")
        significance = "✅ Significant" if test_results['p_value'] < 0.05 else "❌ Not Significant"
        
        st.markdown(f"""
        **Test Type:** {test_results['test_type']}  
        **Test Statistic:** {test_results['statistic']:.4f}  
        **P-value:** {test_results['p_value']:.2e}  
        **Significance:** {significance}  
        **Effect Size:** {test_results['effect_size']:.4f}  
        **Cohen's d/h:** {test_results['cohens_d']:.4f}  
        **Statistical Power:** {power:.4f}  
        **Sample Size (Control):** {test_results['sample_size_control']:,}  
        **Sample Size (Treatment):** {test_results['sample_size_treatment']:,}  
        """)
        
        st.markdown(f"""
        **95% Confidence Interval:**  
        [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}]
        """)
        
        # Skewness analysis
        if selected_metric != 'order_completed':
            combined_data = np.concatenate([control_data, treatment_data])
            skewness_val = skew(combined_data[~np.isnan(combined_data)])
            st.markdown(f"""
            **Distribution Analysis:**  
            Skewness: {skewness_val:.3f}  
            Test Choice: {'Non-parametric' if abs(skewness_val) > 2 else 'Parametric'}
            """)
    
    with col2:
        st.markdown("#### Power Analysis")
        
        # Power curve
        sample_sizes = np.logspace(2, 5, 50)
        powers = []
        
        for n in sample_sizes:
            power_n = calculate_statistical_power(test_results['cohens_d'], int(n/2), int(n/2))
            powers.append(power_n)
        
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines', name='Power Curve'))
        fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                           annotation_text="80% Power Threshold")
        fig_power.add_vline(x=len(control_data) + len(treatment_data), line_dash="dash", 
                           line_color="green", annotation_text="Actual Sample Size")
        
        fig_power.update_layout(
            title="Statistical Power Curve",
            xaxis_title="Total Sample Size",
            yaxis_title="Statistical Power",
            xaxis_type="log"
        )
        st.plotly_chart(fig_power, use_container_width=True)
    
    # Hypothesis testing visualization
    st.subheader("🎯 Hypothesis Testing Visualization")
    
    if selected_metric != 'order_completed':
        se = np.sqrt(np.var(control_data, ddof=1)/len(control_data) + 
                    np.var(treatment_data, ddof=1)/len(treatment_data))
        x_range = np.linspace(-4*se, test_results['effect_size'] + 4*se, 1000)
        
        null_dist = stats.norm.pdf(x_range, 0, se)
        alt_dist = stats.norm.pdf(x_range, test_results['effect_size'], se)
        
        fig_hyp = go.Figure()
        fig_hyp.add_trace(go.Scatter(x=x_range, y=null_dist, mode='lines', 
                                   name='Null Hypothesis (H₀)', fill='tonexty'))
        fig_hyp.add_trace(go.Scatter(x=x_range, y=alt_dist, mode='lines', 
                                   name='Alternative Hypothesis (H₁)', fill='tonexty'))
        fig_hyp.add_vline(x=test_results['effect_size'], line_dash="dash", line_color="red", 
                         annotation_text=f"Observed Effect: {test_results['effect_size']:.4f}")
        
        fig_hyp.update_layout(
            title="Null vs Alternative Hypothesis Distributions",
            xaxis_title="Effect Size",
            yaxis_title="Density"
        )
        st.plotly_chart(fig_hyp, use_container_width=True)

elif analysis_type == "Practical Significance":
    st.subheader("⚡ Practical Significance Analysis")
    
    practical_threshold = config.MIN_PRACTICAL_SIGNIFICANCE.get(selected_metric, 0)
    is_practically_significant = abs(test_results['effect_size']) >= practical_threshold
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Effect Size", f"{test_results['effect_size']:.4f}")
    
    with col2:
        st.metric("Practical Threshold", f"{practical_threshold:.4f}")
    
    with col3:
        status = "✅ Practically Significant" if is_practically_significant else "❌ Not Practically Significant"
        st.metric("Practical Significance", status)
    
    # Practical significance visualization
    fig_practical = go.Figure()
    
    # Add threshold bands
    fig_practical.add_hrect(y0=-practical_threshold, y1=practical_threshold, 
                           fillcolor="red", opacity=0.2, 
                           annotation_text="Not Practically Significant")
    
    if test_results['effect_size'] > 0:
        fig_practical.add_hrect(y0=practical_threshold, y1=max(test_results['ci_upper'], practical_threshold*2), 
                               fillcolor="green", opacity=0.2, 
                               annotation_text="Practically Significant")
    else:
        fig_practical.add_hrect(y0=min(test_results['ci_lower'], -practical_threshold*2), y1=-practical_threshold, 
                               fillcolor="green", opacity=0.2, 
                               annotation_text="Practically Significant")
    
    # Add effect size with confidence interval
    fig_practical.add_trace(go.Scatter(
        x=[1], y=[test_results['effect_size']], 
        error_y=dict(
            type='data',
            array=[test_results['ci_upper'] - test_results['effect_size']],
            arrayminus=[test_results['effect_size'] - test_results['ci_lower']]
        ),
        mode='markers',
        marker=dict(size=15, color='blue'),
        name='Observed Effect'
    ))
    
    fig_practical.add_hline(y=0, line_dash="dash", line_color="black")
    fig_practical.add_hline(y=practical_threshold, line_dash="dot", line_color="red")
    fig_practical.add_hline(y=-practical_threshold, line_dash="dot", line_color="red")
    
    fig_practical.update_layout(
        title="Practical Significance Assessment",
        xaxis_title="",
        yaxis_title="Effect Size",
        xaxis=dict(showticklabels=False, range=[0.5, 1.5])
    )
    st.plotly_chart(fig_practical, use_container_width=True)
    
    # Business context
    st.subheader("📊 Business Context")
    
    if selected_metric == "satisfaction_score":
        st.markdown(f"""
        **Satisfaction Score Analysis:**
        - Minimum meaningful improvement: {practical_threshold} points
        - Observed improvement: {test_results['effect_size']:.3f} points  
        - Business impact: {'Meaningful user experience improvement' if is_practically_significant else 'Minimal user experience impact'}
        """)
    elif selected_metric == "order_value_usd":
        annual_impact = test_results['effect_size'] * 28499 * 365 if is_practically_significant else 0
        st.markdown(f"""
        **Order Value Analysis:**
        - Minimum meaningful improvement: ${practical_threshold} per order
        - Observed improvement: ${test_results['effect_size']:.2f} per order
        - Estimated annual revenue impact: ${annual_impact:,.0f}
        """)
    elif selected_metric == "order_completed":
        conversion_impact = test_results['effect_size'] * 28499 * 365 if is_practically_significant else 0
        st.markdown(f"""
        **Conversion Rate Analysis:**
        - Minimum meaningful improvement: {practical_threshold:.1%}
        - Observed improvement: {test_results['effect_size']:.1%}
        - Estimated additional annual orders: {conversion_impact:,.0f}
        """)

elif analysis_type == "Effect Size Analysis":
    st.subheader("📏 Effect Size Analysis")
    
    # Effect size interpretation
    cohens_d_abs = abs(test_results['cohens_d'])
    
    if cohens_d_abs < 0.2:
        effect_magnitude = "Negligible"
        color = "red"
    elif cohens_d_abs < 0.5:
        effect_magnitude = "Small"
        color = "orange"
    elif cohens_d_abs < 0.8:
        effect_magnitude = "Medium"
        color = "yellow"
    else:
        effect_magnitude = "Large"
        color = "green"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cohen's d", f"{test_results['cohens_d']:.4f}")
    
    with col2:
        st.metric("Effect Magnitude", effect_magnitude)
    
    with col3:
        st.metric("Relative Lift", f"{relative_lift:.2f}%")
    
    # Effect size benchmarks
    fig_effect = go.Figure()
    
    benchmarks = ['Negligible\n(< 0.2)', 'Small\n(0.2-0.5)', 'Medium\n(0.5-0.8)', 'Large\n(> 0.8)', 'Observed']
    values = [0.1, 0.35, 0.65, 0.9, cohens_d_abs]
    colors = ['lightgray', 'lightblue', 'orange', 'lightgreen', color]
    
    fig_effect.add_trace(go.Bar(x=benchmarks, y=values, marker_color=colors))
    fig_effect.update_layout(
        title="Effect Size Benchmarks",
        xaxis_title="Effect Size Category",
        yaxis_title="Cohen's d"
    )
    st.plotly_chart(fig_effect, use_container_width=True)
    
    # Effect size across different metrics comparison
    st.subheader("📊 Cross-Metric Effect Size Comparison")
    
    all_metrics = ['satisfaction_score', 'order_completed', 'order_value_usd']
    all_effects = []
    
    for metric in all_metrics:
        metric_control = df[df['group'] == 'control'][metric].values
        metric_treatment = df[df['group'] == 'treatment'][metric].values
        
        if metric == 'order_completed':
            control_successes = metric_control.sum()
            control_total = len(metric_control)
            treatment_successes = metric_treatment.sum()
            treatment_total = len(metric_treatment)
            result = proportion_test_with_ci(control_successes, control_total, 
                                           treatment_successes, treatment_total)
        else:
            combined = np.concatenate([metric_control, metric_treatment])
            skewness_val = abs(skew(combined[~np.isnan(combined)]))
            
            if skewness_val > 2:
                result = mann_whitney_test_with_ci(metric_control, metric_treatment)
            else:
                result = welch_ttest_with_ci(metric_control, metric_treatment)
        
        all_effects.append({
            'metric': metric.replace('_', ' ').title(),
            'cohens_d': result['cohens_d'],
            'p_value': result['p_value'],
            'significant': result['p_value'] < 0.05
        })
    
    effect_df = pd.DataFrame(all_effects)
    
    fig_comparison = px.bar(effect_df, x='metric', y='cohens_d', 
                           color='significant',
                           title='Effect Sizes Across Primary Metrics',
                           color_discrete_map={True: 'green', False: 'red'})
    
    fig_comparison.add_hline(y=0.2, line_dash="dash", annotation_text="Small Effect")
    fig_comparison.add_hline(y=0.5, line_dash="dash", annotation_text="Medium Effect") 
    fig_comparison.add_hline(y=0.8, line_dash="dash", annotation_text="Large Effect")
    
    st.plotly_chart(fig_comparison, use_container_width=True)

elif analysis_type == "Sequential Testing":
    st.subheader("📈 Sequential Testing Analysis")
    
    # Simulate sequential testing
    df_sorted = df.sort_values('order_timestamp').reset_index(drop=True)
    
    n_steps = 50
    sample_sizes = np.linspace(1000, len(df), n_steps).astype(int)
    p_values = []
    effect_sizes = []
    confidence_intervals = []
    
    for n in sample_sizes:
        # Take first n samples
        subset_df = df_sorted.head(n)
        control_subset = subset_df[subset_df['group'] == 'control'][selected_metric].values
        treatment_subset = subset_df[subset_df['group'] == 'treatment'][selected_metric].values
        
        if len(control_subset) > 10 and len(treatment_subset) > 10:
            if selected_metric == 'order_completed':
                control_successes = control_subset.sum()
                control_total = len(control_subset)
                treatment_successes = treatment_subset.sum()
                treatment_total = len(treatment_subset)
                
                result = proportion_test_with_ci(control_successes, control_total,
                                               treatment_successes, treatment_total)
            else:
                combined = np.concatenate([control_subset, treatment_subset])
                skewness_val = abs(skew(combined[~np.isnan(combined)]))
                
                if skewness_val > 2:
                    result = mann_whitney_test_with_ci(control_subset, treatment_subset)
                else:
                    result = welch_ttest_with_ci(control_subset, treatment_subset)
            
            p_values.append(result['p_value'])
            effect_sizes.append(result['effect_size'])
            confidence_intervals.append([result['ci_lower'], result['ci_upper']])
        else:
            p_values.append(1.0)
            effect_sizes.append(0.0)
            confidence_intervals.append([0, 0])
    
    # Sequential testing visualization
    fig_seq = make_subplots(rows=3, cols=1, 
                           subplot_titles=("P-value Evolution", "Effect Size Evolution", "Statistical Power Evolution"),
                           vertical_spacing=0.1)
    
    # P-value evolution
    fig_seq.add_trace(go.Scatter(x=sample_sizes, y=p_values, mode='lines+markers', 
                                name='P-value', line=dict(color='blue')), row=1, col=1)
    fig_seq.add_hline(y=0.05, line_dash="dash", line_color="red", row=1, col=1,
                     annotation_text="α = 0.05")
    
    # Effect size evolution with confidence intervals
    ci_lower = [ci[0] for ci in confidence_intervals]
    ci_upper = [ci[1] for ci in confidence_intervals]
    
    fig_seq.add_trace(go.Scatter(x=sample_sizes, y=effect_sizes, mode='lines+markers',
                                name='Effect Size', line=dict(color='green')), row=2, col=1)
    fig_seq.add_trace(go.Scatter(x=sample_sizes + sample_sizes[::-1], 
                                y=ci_upper + ci_lower[::-1],
                                fill='tonexty', fillcolor='rgba(0,255,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% CI', showlegend=False), row=2, col=1)
    
    # Power evolution
    powers = []
    for i, n in enumerate(sample_sizes):
        if len(effect_sizes) > i and effect_sizes[i] != 0:
            power_val = calculate_statistical_power(effect_sizes[i]/np.std(control_data), 
                                                   int(n/2), int(n/2))
            powers.append(power_val)
        else:
            powers.append(0)
    
    fig_seq.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines+markers',
                                name='Statistical Power', line=dict(color='orange')), row=3, col=1)
    fig_seq.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=1,
                     annotation_text="Desired Power = 0.8")
    
    fig_seq.update_xaxes(title_text="Sample Size", row=3, col=1)
    fig_seq.update_yaxes(title_text="P-value", row=1, col=1)
    fig_seq.update_yaxes(title_text="Effect Size", row=2, col=1)
    fig_seq.update_yaxes(title_text="Statistical Power", row=3, col=1)
    fig_seq.update_layout(height=800, title_text="Sequential Testing Results")
    
    st.plotly_chart(fig_seq, use_container_width=True)
    
    # Sequential decision points
    st.subheader("🚦 Sequential Decision Analysis")
    first_significant = next((i for i, p in enumerate(p_values) if p < 0.05), None)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if first_significant is not None:
            st.metric("First Significant Result", f"{sample_sizes[first_significant]:,} samples")
        else:
            st.metric("First Significant Result", "Not detected")
    
    with col2:
        if first_significant is not None:
            savings = len(df) - sample_sizes[first_significant]
            st.metric("Potential Sample Savings", f"{savings:,} samples")
        else:
            st.metric("Potential Sample Savings", "N/A")
    
    with col3:
        if first_significant is not None:
            savings_pct = (len(df) - sample_sizes[first_significant]) / len(df) * 100
            st.metric("Savings Percentage", f"{savings_pct:.1f}%")
        else:
            st.metric("Savings Percentage", "N/A")

elif analysis_type == "Business Impact":
    st.subheader("💰 Business Impact Assessment")
    
    # Business parameters
    total_orders = len(df)
    test_duration = 14  # days
    daily_orders = total_orders / test_duration
    unique_users = 100000  # From original data
    estimated_daily_unique_users = daily_orders * 0.54  # Assumption from original analysis
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Test Orders", f"{total_orders:,}")
    
    with col2:
        st.metric("Daily Orders", f"{daily_orders:,.0f}")
    
    with col3:
        st.metric("Unique Users", f"{unique_users:,}")
    
    with col4:
        st.metric("Est. Daily Active Users", f"{estimated_daily_unique_users:,.0f}")
    
    # Revenue impact analysis
    if selected_metric == "order_value_usd" and test_results['p_value'] < 0.05:
        st.subheader("📈 Revenue Impact Analysis")
        
        aov_improvement = test_results['effect_size']
        daily_revenue_impact = estimated_daily_unique_users * aov_improvement
        monthly_revenue_impact = daily_revenue_impact * 30
        annual_revenue_impact = daily_revenue_impact * 365
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AOV Improvement", f"${aov_improvement:.2f}", 
                     f"{relative_lift:.1f}%")
        
        with col2:
            st.metric("Monthly Revenue Impact", f"${monthly_revenue_impact:,.0f}")
        
        with col3:
            st.metric("Annual Revenue Impact", f"${annual_revenue_impact:,.0f}")
        
        # Revenue projection visualization
        months = np.arange(1, 13)
        cumulative_revenue = months * (annual_revenue_impact / 12)
        
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Scatter(x=months, y=cumulative_revenue, 
                                       mode='lines+markers', name='Cumulative Revenue Impact',
                                       fill='tonexty'))
        fig_revenue.update_layout(
            title="Projected Annual Revenue Impact",
            xaxis_title="Month",
            yaxis_title="Cumulative Revenue Impact ($)"
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
    elif selected_metric == "order_completed" and test_results['p_value'] < 0.05:
        st.subheader("🎯 Conversion Impact Analysis")
        
        conversion_improvement = test_results['effect_size']
        additional_daily_orders = estimated_daily_unique_users * conversion_improvement
        additional_annual_orders = additional_daily_orders * 365
        avg_order_value = 30.10  # From original data
        additional_annual_revenue = additional_annual_orders * avg_order_value
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conversion Improvement", f"{conversion_improvement:.1%}", 
                     f"{relative_lift:.1f}%")
        
        with col2:
            st.metric("Additional Annual Orders", f"{additional_annual_orders:,.0f}")
        
        with col3:
            st.metric("Additional Annual Revenue", f"${additional_annual_revenue:,.0f}")
        
    elif selected_metric == "satisfaction_score" and test_results['p_value'] < 0.05:
        st.subheader("😊 User Experience Impact")
        
        satisfaction_improvement = test_results['effect_size']
        # Estimate retention impact (hypothetical relationship)
        retention_impact = satisfaction_improvement * 0.2  # 20% of satisfaction improvement
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Satisfaction Improvement", f"{satisfaction_improvement:.3f}", 
                     f"{relative_lift:.1f}%")
        
        with col2:
            st.metric("Estimated Retention Impact", f"+{retention_impact:.1%}")
    
    # ROI Analysis
    st.subheader("📊 Return on Investment Analysis")
    
    # Hypothetical costs
    implementation_cost = 500000  # $500K
    monthly_maintenance = 50000   # $50K/month
    
    if selected_metric == "order_value_usd" and test_results['p_value'] < 0.05:
        monthly_revenue_benefit = annual_revenue_impact / 12
        net_monthly_benefit = monthly_revenue_benefit - monthly_maintenance
        payback_months = implementation_cost / net_monthly_benefit if net_monthly_benefit > 0 else float('inf')
        annual_roi = ((annual_revenue_impact - 12*monthly_maintenance - implementation_cost) / implementation_cost) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Implementation Cost", f"${implementation_cost:,}")
        
        with col2:
            st.metric("Monthly Maintenance", f"${monthly_maintenance:,}")
        
        with col3:
            if payback_months != float('inf'):
                st.metric("Payback Period", f"{payback_months:.1f} months")
            else:
                st.metric("Payback Period", "Not profitable")
        
        with col4:
            st.metric("Annual ROI", f"{annual_roi:.1f}%")

elif analysis_type == "Segmentation":
    st.subheader("🎯 Advanced Segmentation Analysis")
    
    # Segment selection
    segment_var = st.selectbox("Select Segmentation Dimension:", 
                              ["city", "age_group", "user_frequency_segment"])
    
    # Calculate segment effects
    segments = df[segment_var].unique()
    segment_results = []
    
    for segment in segments:
        segment_data = df[df[segment_var] == segment]
        control_seg = segment_data[segment_data['group'] == 'control'][selected_metric].values
        treatment_seg = segment_data[segment_data['group'] == 'treatment'][selected_metric].values
        
        if len(control_seg) > 10 and len(treatment_seg) > 10:
            if selected_metric == 'order_completed':
                control_successes = control_seg.sum()
                control_total = len(control_seg)
                treatment_successes = treatment_seg.sum()
                treatment_total = len(treatment_seg)
                
                result = proportion_test_with_ci(control_successes, control_total,
                                               treatment_successes, treatment_total)
            else:
                combined = np.concatenate([control_seg, treatment_seg])
                skewness_val = abs(skew(combined[~np.isnan(combined)]))
                
                if skewness_val > 2:
                    result = mann_whitney_test_with_ci(control_seg, treatment_seg)
                else:
                    result = welch_ttest_with_ci(control_seg, treatment_seg)
            
            relative_lift_seg = (result['effect_size'] / result['control_mean']) * 100 if result['control_mean'] != 0 else 0
            
            segment_results.append({
                'Segment': segment,
                'Control Mean': result['control_mean'],
                'Treatment Mean': result['treatment_mean'],
                'Effect Size': result['effect_size'],
                'Relative Lift (%)': relative_lift_seg,
                'P-value': result['p_value'],
                'Cohens d': result['cohens_d'],
                'Sample Size': len(control_seg) + len(treatment_seg),
                'Significant': result['p_value'] < 0.05
            })
    
    # Display results
    if segment_results:
        segment_df = pd.DataFrame(segment_results)
        
        # Visualization
        fig_seg = px.bar(segment_df, x='Segment', y='Relative Lift (%)', 
                        title=f'Relative Lift by {segment_var.replace("_", " ").title()}',
                        color='Significant', 
                        color_discrete_map={True: 'green', False: 'red'},
                        hover_data=['Effect Size', 'P-value', 'Sample Size'])
        
        fig_seg.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_seg, use_container_width=True)
        
        # Effect size comparison
        fig_effect_seg = px.bar(segment_df, x='Segment', y='Cohens d',
                               title=f'Effect Size by {segment_var.replace("_", " ").title()}',
                               color='Significant',
                               color_discrete_map={True: 'green', False: 'red'})
        
        fig_effect_seg.add_hline(y=0.2, line_dash="dash", annotation_text="Small Effect")
        fig_effect_seg.add_hline(y=0.5, line_dash="dash", annotation_text="Medium Effect")
        fig_effect_seg.add_hline(y=0.8, line_dash="dash", annotation_text="Large Effect")
        
        st.plotly_chart(fig_effect_seg, use_container_width=True)
        
        # Detailed segment table
        st.subheader("📋 Detailed Segment Analysis")
        
        # Format the dataframe for better display
        display_df = segment_df.copy()
        for col in ['Control Mean', 'Treatment Mean', 'Effect Size', 'Cohens d']:
            display_df[col] = display_df[col].round(4)
        display_df['Relative Lift (%)'] = display_df['Relative Lift (%)'].round(2)
        display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f}")
        display_df['Significant'] = display_df['Significant'].map({True: "✅", False: "❌"})
        
        st.dataframe(display_df, use_container_width=True)
        
        # Top and bottom performers
        significant_segments = segment_df[segment_df['Significant'] == True]
        
        if not significant_segments.empty:
            st.subheader("🏆 Segment Performance Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Performing Segments:**")
                top_segments = significant_segments.nlargest(3, 'Relative Lift (%)')
                for _, row in top_segments.iterrows():
                    st.success(f"**{row['Segment']}**: {row['Relative Lift (%)']:.2f}% lift")
            
            with col2:
                st.markdown("**Segments Needing Attention:**")
                if len(significant_segments[significant_segments['Relative Lift (%)'] < 0]) > 0:
                    bottom_segments = significant_segments.nsmallest(3, 'Relative Lift (%)')
                    for _, row in bottom_segments.iterrows():
                        st.error(f"**{row['Segment']}**: {row['Relative Lift (%)']:.2f}% change")
                else:
                    st.info("All significant segments show positive performance")

elif analysis_type == "Data Quality":
    st.subheader("🔍 Comprehensive Data Quality Analysis")
    
    # Sample ratio analysis
    st.subheader("⚖️ Sample Ratio Analysis")
    
    control_ratio = len(df[df['group'] == 'control']) / len(df)
    treatment_ratio = len(df[df['group'] == 'treatment']) / len(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Control Ratio", f"{control_ratio:.1%}")
    
    with col2:
        st.metric("Treatment Ratio", f"{treatment_ratio:.1%}")
    
    with col3:
        balance_check = "✅ Balanced" if abs(control_ratio - 0.5) < 0.02 else "⚠️ Imbalanced"
        st.metric("Balance Check", balance_check)
    
    # Outlier detection
    st.subheader("📊 Outlier Analysis")
    
    def detect_outliers_iqr(data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)
    
    def detect_outliers_zscore(data):
        z_scores = np.abs(stats.zscore(data))
        return z_scores > 3
    
    control_outliers_iqr = detect_outliers_iqr(control_data)
    treatment_outliers_iqr = detect_outliers_iqr(treatment_data)
    control_outliers_z = detect_outliers_zscore(control_data)
    treatment_outliers_z = detect_outliers_zscore(treatment_data)
    
    outlier_summary = pd.DataFrame({
        'Group': ['Control', 'Treatment', 'Control', 'Treatment'],
        'Method': ['IQR', 'IQR', 'Z-Score', 'Z-Score'],
        'Total Samples': [len(control_data), len(treatment_data), len(control_data), len(treatment_data)],
        'Outliers': [control_outliers_iqr.sum(), treatment_outliers_iqr.sum(), 
                    control_outliers_z.sum(), treatment_outliers_z.sum()],
        'Outlier Rate': [control_outliers_iqr.mean(), treatment_outliers_iqr.mean(),
                        control_outliers_z.mean(), treatment_outliers_z.mean()]
    })
    
    st.dataframe(outlier_summary, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📈 Distribution Analysis")
    
    # Normality tests
    if len(control_data) <= 5000:
        control_shapiro_stat, control_shapiro_p = shapiro(control_data[:5000])
        treatment_shapiro_stat, treatment_shapiro_p = shapiro(treatment_data[:5000])
        
        normality_results = pd.DataFrame({
            'Group': ['Control', 'Treatment'],
            'Shapiro-Wilk Statistic': [control_shapiro_stat, treatment_shapiro_stat],
            'Shapiro-Wilk P-value': [control_shapiro_p, treatment_shapiro_p],
            'Normal Distribution': [
                "✅ Yes" if control_shapiro_p > 0.05 else "❌ No",
                "✅ Yes" if treatment_shapiro_p > 0.05 else "❌ No"
            ]
        })
        
        st.dataframe(normality_results, use_container_width=True)
    
    # Distribution statistics
    dist_stats = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
        'Control': [np.mean(control_data), np.median(control_data), np.std(control_data),
                   skew(control_data), kurtosis(control_data), np.min(control_data), np.max(control_data)],
        'Treatment': [np.mean(treatment_data), np.median(treatment_data), np.std(treatment_data),
                     skew(treatment_data), kurtosis(treatment_data), np.min(treatment_data), np.max(treatment_data)]
    })
    
    st.dataframe(dist_stats, use_container_width=True)
    
    # Missing data analysis
    st.subheader("🔍 Missing Data Analysis")
    
    missing_analysis = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': [df[col].isnull().sum() for col in df.columns],
        'Missing Rate': [df[col].isnull().mean() for col in df.columns]
    })
    
    missing_analysis = missing_analysis[missing_analysis['Missing Count'] > 0]
    
    if not missing_analysis.empty:
        st.dataframe(missing_analysis, use_container_width=True)
    else:
        st.success("✅ No missing data detected in the dataset")

# Additional advanced features
st.sidebar.markdown("---")
st.sidebar.header("🔍 Advanced Features")

# Recommendation system analysis
if st.sidebar.checkbox("Show Recommendation System Analysis"):
    st.subheader("🤖 Smart Recommendations System Performance")
    
    treatment_data = df[df['group'] == 'treatment']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        adoption_rate = (treatment_data['recommendations_shown'] > 0).mean()
        st.metric("Adoption Rate", f"{adoption_rate:.1%}")
    
    with col2:
        avg_recs_shown = treatment_data['recommendations_shown'].mean()
        st.metric("Avg Recommendations", f"{avg_recs_shown:.1f}")
    
    with col3:
        ctr = (treatment_data['recommendations_clicked'] > 0).mean()
        st.metric("Click-Through Rate", f"{ctr:.1%}")
    
    with col4:
        conversion_rate = treatment_data['ordered_from_recommendation'].mean()
        st.metric("Conversion Rate", f"{conversion_rate:.1%}")
    
    # Recommendation funnel
    st.subheader("📊 Recommendation Conversion Funnel")
    
    total_users = len(treatment_data)
    users_with_recs = (treatment_data['recommendations_shown'] > 0).sum()
    users_clicked = (treatment_data['recommendations_clicked'] > 0).sum()
    users_converted = treatment_data['ordered_from_recommendation'].sum()
    
    funnel_data = {
        'Stage': ['Total Users', 'Received Recommendations', 'Clicked Recommendations', 'Ordered from Recommendations'],
        'Users': [total_users, users_with_recs, users_clicked, users_converted],
        'Conversion Rate': [100, 
                           (users_with_recs/total_users)*100,
                           (users_clicked/users_with_recs)*100 if users_with_recs > 0 else 0,
                           (users_converted/users_clicked)*100 if users_clicked > 0 else 0]
    }
    
    fig_funnel = go.Figure()
    fig_funnel.add_trace(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Users'],
        texttemplate="%{value:,}<br>(%{percentInitial})",
        textposition="inside"
    ))
    
    fig_funnel.update_layout(title="Recommendation System Funnel Analysis")
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # Feature correlation analysis
    st.subheader("🔗 Feature Correlation Analysis")
    
    rec_metrics = ['recommendations_shown', 'recommendations_clicked', 'ordered_from_recommendation', 
                   'satisfaction_score', 'order_value_usd', 'items_viewed']
    
    corr_matrix = treatment_data[rec_metrics].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        title="Recommendation Feature Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        text_auto=True)
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Executive summary generator
if st.sidebar.button("📊 Generate Executive Summary"):
    st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
    st.subheader("📋 Executive Summary Report")
    
    # Calculate results for all primary metrics
    all_metrics = ['satisfaction_score', 'order_completed', 'order_value_usd']
    summary_results = {}
    
    for metric in all_metrics:
        control_m = df[df['group'] == 'control'][metric].values
        treatment_m = df[df['group'] == 'treatment'][metric].values
        
        if metric == 'order_completed':
            control_successes = control_m.sum()
            control_total = len(control_m)
            treatment_successes = treatment_m.sum()
            treatment_total = len(treatment_m)
            
            result = proportion_test_with_ci(control_successes, control_total,
                                           treatment_successes, treatment_total)
        else:
            combined = np.concatenate([control_m, treatment_m])
            skewness_val = abs(skew(combined[~np.isnan(combined)]))
            
            if skewness_val > 2:
                result = mann_whitney_test_with_ci(control_m, treatment_m)
            else:
                result = welch_ttest_with_ci(control_m, treatment_m)
        
        summary_results[metric] = {
            'control_mean': result['control_mean'],
            'treatment_mean': result['treatment_mean'],
            'effect_size': result['effect_size'],
            'relative_lift': (result['effect_size'] / result['control_mean']) * 100 if result['control_mean'] != 0 else 0,
            'p_value': result['p_value'],
            'significant': result['p_value'] < 0.05,
            'cohens_d': result['cohens_d']
        }
    
    st.markdown(f"""
    ## 🎯 Experiment: Smart Recommendations Feature
    **Duration:** October 1-15, 2024 (14 days)  
    **Sample Size:** {len(df):,} total orders  
    **Users:** 100,000 unique users  
    **Test Design:** Two-group randomized controlled trial
    
    ### 📊 Primary Metrics Results
    """)
    
    # Results table
    results_df = pd.DataFrame(summary_results).T
    results_df.index = ['Satisfaction Score', 'Order Completion Rate', 'Average Order Value']
    results_df['Significant'] = results_df['significant'].map({True: "✅", False: "❌"})
    
    # Format for display
    display_results = results_df.copy()
    for col in ['control_mean', 'treatment_mean', 'effect_size', 'cohens_d']:
        display_results[col] = display_results[col].round(4)
    display_results['relative_lift'] = display_results['relative_lift'].round(2)
    display_results['p_value'] = display_results['p_value'].apply(lambda x: f"{x:.2e}")
    
    st.dataframe(display_results[['control_mean', 'treatment_mean', 'effect_size', 
                                'relative_lift', 'p_value', 'cohens_d', 'Significant']], 
                use_container_width=True)
    
    # Business impact summary
    significant_count = sum(summary_results[m]['significant'] for m in all_metrics)
    
    st.markdown(f"""
    ### 💰 Business Impact Summary
    **Success Rate:** {significant_count}/3 primary metrics significant  
    **Satisfaction Improvement:** +{summary_results['satisfaction_score']['relative_lift']:.1f}%  
    **Conversion Improvement:** +{summary_results['order_completed']['relative_lift']:.1f}%  
    **AOV Improvement:** +{summary_results['order_value_usd']['relative_lift']:.1f}%  
    
    ### 🔬 Statistical Rigor Applied
    - Welch's t-tests for continuous metrics (robust to unequal variances)
    - Mann-Whitney U tests for highly skewed distributions
    - Proportion tests with proper confidence intervals
    - Cohen's d/h effect size calculations
    - Bootstrap confidence intervals where appropriate
    - Statistical power analysis
    - Sequential testing simulation
    - Comprehensive segmentation analysis
    
    ### 🚀 Recommendation
    """)
    
    if significant_count >= 2:
        st.success("✅ **LAUNCH RECOMMENDED** - Feature shows strong positive impact across key metrics")
        
        # Calculate estimated annual impact
        if summary_results['order_value_usd']['significant']:
            aov_impact = summary_results['order_value_usd']['effect_size']
            estimated_annual_revenue = aov_impact * 28499 * 365  # Daily orders * days
            st.markdown(f"**Estimated Annual Revenue Impact:** ${estimated_annual_revenue:,.0f}")
        
        if summary_results['order_completed']['significant']:
            conversion_impact = summary_results['order_completed']['effect_size']
            additional_orders = conversion_impact * 28499 * 365
            st.markdown(f"**Additional Annual Orders:** {additional_orders:,.0f}")
            
    elif significant_count == 1:
        st.warning("⚠️ **PROCEED WITH CAUTION** - Mixed results, consider additional testing")
    else:
        st.error("❌ **DO NOT LAUNCH** - No significant improvements detected")
    
    st.markdown("""
    ### 📋 Next Steps
    1. 🚀 Prepare for full rollout if launching
    2. 📊 Set up continuous monitoring dashboard  
    3. 🎯 Implement automated alerting system
    4. 📈 Plan follow-up analysis in 30-60 days
    5. 🧪 Design iteration experiments for optimization
    
    ### ⚠️ Risk Assessment
    """)
    
    # Risk assessment
    risks = []
    
    # Check for negative significant effects
    negative_effects = [m for m in all_metrics if summary_results[m]['significant'] and summary_results[m]['effect_size'] < 0]
    if negative_effects:
        risks.append(f"Significant negative impact detected in: {', '.join(negative_effects)}")
    
    # Check for low statistical power
    low_power_metrics = [m for m in all_metrics if abs(summary_results[m]['cohens_d']) < 0.2]
    if low_power_metrics:
        risks.append(f"Small effect sizes in: {', '.join(low_power_metrics)}")
    
    # Check practical significance
    practical_issues = []
    for metric in all_metrics:
        threshold = config.MIN_PRACTICAL_SIGNIFICANCE.get(metric, 0)
        if abs(summary_results[metric]['effect_size']) < threshold:
            practical_issues.append(metric)
    
    if practical_issues:
        risks.append(f"Effects below practical significance threshold: {', '.join(practical_issues)}")
    
    if not risks:
        st.success("✅ No significant risks identified")
    else:
        for i, risk in enumerate(risks, 1):
            st.error(f"{i}. {risk}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Time series analysis
if st.sidebar.checkbox("Show Time Series Analysis"):
    st.subheader("📈 Time Series Analysis")
    
    # Generate synthetic daily data
    dates = pd.date_range('2024-10-01', '2024-10-15', freq='D')
    daily_metrics = []
    
    for i, date in enumerate(dates):
        # Simulate daily variation
        daily_sample_size = len(df) // 14  # Roughly equal daily samples
        daily_control = df[df['group'] == 'control'].sample(n=min(daily_sample_size//2, len(df[df['group'] == 'control'])), random_state=i)
        daily_treatment = df[df['group'] == 'treatment'].sample(n=min(daily_sample_size//2, len(df[df['group'] == 'treatment'])), random_state=i)
        
        daily_metrics.append({
            'date': date,
            'control_mean': daily_control[selected_metric].mean(),
            'treatment_mean': daily_treatment[selected_metric].mean(),
            'control_volume': len(daily_control),
            'treatment_volume': len(daily_treatment)
        })
    
    ts_df = pd.DataFrame(daily_metrics)
    ts_df['lift'] = (ts_df['treatment_mean'] - ts_df['control_mean']) / ts_df['control_mean'] * 100
    ts_df['absolute_effect'] = ts_df['treatment_mean'] - ts_df['control_mean']
    
    # Time series visualization
    fig_ts = make_subplots(rows=3, cols=1, 
                          subplot_titles=(f"Daily {selected_metric.replace('_', ' ').title()}", 
                                        "Daily Lift Percentage", "Daily Volume"),
                          vertical_spacing=0.1)
    
    # Daily means
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['control_mean'], 
                               name='Control', mode='lines+markers'), row=1, col=1)
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['treatment_mean'], 
                               name='Treatment', mode='lines+markers'), row=1, col=1)
    
    # Daily lift
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['lift'], 
                               name='Lift %', mode='lines+markers', 
                               line=dict(color='green')), row=2, col=1)
    fig_ts.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    # Daily volume
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['control_volume'], 
                               name='Control Volume', mode='lines+markers'), row=3, col=1)
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['treatment_volume'], 
                               name='Treatment Volume', mode='lines+markers'), row=3, col=1)
    
    fig_ts.update_layout(height=800, title_text="Time Series Analysis")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Time series statistics
    st.subheader("📊 Time Series Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_lift = ts_df['lift'].mean()
        st.metric("Average Daily Lift", f"{avg_lift:.2f}%")
    
    with col2:
        lift_stability = ts_df['lift'].std()
        st.metric("Lift Stability (Std Dev)", f"{lift_stability:.2f}%")
    
    with col3:
        positive_days = (ts_df['lift'] > 0).sum()
        st.metric("Positive Days", f"{positive_days}/{len(ts_df)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    📊 Uber Eats A/B Test Advanced Analytics Dashboard | Built with Streamlit & Plotly<br>
    Statistical Framework: Welch's t-tests, Mann-Whitney U, Bootstrap CI, Sequential Testing
</div>
""", unsafe_allow_html=True)
    
    #
