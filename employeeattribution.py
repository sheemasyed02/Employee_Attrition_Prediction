import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero')
 
st.set_page_config(
    page_title="Workforce Analytics Suite", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS layout structure
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    
    .enterprise-header {
        background-color: #2c3e50;
        color: white;
        padding: 25px;
        border-radius: 8px;
        margin-bottom: 25px;
        text-align: left;
        border-left: 6px solid #3498db;
    }
    
    .analytics-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 15px;
        color: #2c3e50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .content-divider {
        background-color: #ecf0f1;
        padding: 12px;
        border-left: 5px solid #3498db;
        margin: 25px 0 15px 0;
        font-weight: bold;
        color: #2c3e50;
        border-radius: 0 4px 4px 0;
    }
    
    .data-panel {
        background-color: #e8f4fd;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 12px 0;
        color: #2c3e50;
    }
    
    .insights-panel {
        background-color: #d4edda;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 12px 0;
        color: #155724;
    }
    
    .alert-panel {
        background-color: #fff3cd;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 12px 0;
        color: #856404;
    }
    
    .visualization-frame {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 12px 0;
        color: #2c3e50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'employeedata.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    
    return df

def preprocess_data(df):
    df_processed = df.copy()
    
    df_processed['Attrition'] = df_processed['Attrition'].str.upper()
    df_processed['Attrition'] = df_processed['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    # Numerical encoding
    label_cols = ['Attrition', 'Gender', 'Over18', 'OverTime']
    for col in label_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

    # One-hot encoding
    categorical_cols = ['BusinessTravel', 'Department', 'MaritalStatus', 'EducationField', 'JobRole']
    existing_categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
    
    if existing_categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=existing_categorical_cols, drop_first=False)

    return df_processed

def executive_summary_report(df):
    # Clean data
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="enterprise-header">
        <h1>Executive Summary Report</h1>
        <p>Strategic workforce overview and key performance indicators</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Key Metrics", "Trends", "Insights"])
    
    with tab1:
        total_employees = len(df_clean)
        attrition_rate = (df_clean['Attrition'] == 'Yes').mean() * 100
        avg_tenure = df_clean['YearsAtCompany'].mean()
        avg_age = df_clean['Age'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #3498db;">
                <h3 style="margin: 0; color: #3498db; font-size: 1.8em;">{total_employees:,}</h3>
                <p style="margin: 8px 0 0 0; color: #7f8c8d; font-size: 0.9em;">Total Employees</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "#e74c3c" if attrition_rate > 15 else "#f39c12" if attrition_rate > 10 else "#27ae60"
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid {color};">
                <h3 style="margin: 0; color: {color}; font-size: 1.8em;">{attrition_rate:.1f}%</h3>
                <p style="margin: 8px 0 0 0; color: #7f8c8d; font-size: 0.9em;">Turnover Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #27ae60;">
                <h3 style="margin: 0; color: #27ae60; font-size: 1.8em;">{avg_tenure:.1f}</h3>
                <p style="margin: 8px 0 0 0; color: #7f8c8d; font-size: 0.9em;">Average Tenure (Years)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #9b59b6;">
                <h3 style="margin: 0; color: #9b59b6; font-size: 1.8em;">{avg_age:.1f}</h3>
                <p style="margin: 8px 0 0 0; color: #7f8c8d; font-size: 0.9em;">Average Age</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'Department' in df_clean.columns:
                dept_count = len(df_clean['Department'].unique())
                active_employees = len(df_clean[df_clean['Attrition'] == 'No'])
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h5 style="margin: 0 0 10px 0; color: #495057;">Workforce Overview</h5>
                    <p style="margin: 5px 0; color: #6c757d;"><strong>Active Employees:</strong> {active_employees:,}</p>
                    <p style="margin: 5px 0; color: #6c757d;"><strong>Departments:</strong> {dept_count}</p>
                    <p style="margin: 5px 0; color: #6c757d;"><strong>Retention Rate:</strong> {100-attrition_rate:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'MonthlyIncome' in df_clean.columns:
                avg_income = df_clean['MonthlyIncome'].mean()
                st.markdown(f"""
                <div style="background: #e8f4fd; padding: 15px; border-radius: 8px; border: 1px solid #bee5eb;">
                    <h5 style="margin: 0 0 10px 0; color: #0c5460;">Compensation</h5>
                    <p style="margin: 5px 0; color: #0c5460;"><strong>Average Income:</strong></p>
                    <p style="margin: 5px 0; color: #0c5460; font-size: 1.2em;"><strong>${avg_income:,.0f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Workforce Status")
            attrition_counts = df_clean['Attrition'].value_counts()
            
            labels = ['Currently Employed' if x == 'No' else 'No Longer With Company' for x in attrition_counts.index]
            colors = ['#2ecc71', '#e74c3c']
            
            fig = px.pie(
                values=attrition_counts.values, 
                names=labels,
                title='Employee Status Distribution',
                color_discrete_sequence=colors,
                hole=0.4  
            )
            
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Age Distribution Trends")
            
            fig = px.histogram(
                df_clean, 
                x='Age', 
                nbins=15,
                title='Employee Age Groups',
                color_discrete_sequence=['#3498db'],
                marginal="box" 
            )
            
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, width='stretch')
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            high_risk = len(df_clean[(df_clean['Attrition'] == 'Yes')])
            retention_rate = 100 - attrition_rate
            
            st.markdown(f"""
            <div style="background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6; margin: 8px 0; border-left: 4px solid #dc3545;">
                <h5 style="color: #dc3545; margin: 0 0 8px 0;">Attrition Analysis</h5>
                <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Employees Lost:</strong> {high_risk}</p>
                <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Retention Rate:</strong> {retention_rate:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'Department' in df_clean.columns:
                worst_dept = df_clean.groupby('Department', observed=True)['Attrition'].apply(lambda x: (x == 'Yes').mean()).idxmax()
                worst_rate = df_clean.groupby('Department', observed=True)['Attrition'].apply(lambda x: (x == 'Yes').mean()).max() * 100
                
                st.markdown(f"""
                <div style="background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6; margin: 8px 0; border-left: 4px solid #ffc107;">
                    <h5 style="color: #856404; margin: 0 0 8px 0;">Department Alert</h5>
                    <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Highest Turnover:</strong> {worst_dept}</p>
                    <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Rate:</strong> {worst_rate:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if 'MonthlyIncome' in df_clean.columns:
                left_avg_income = df_clean[df_clean['Attrition'] == 'Yes']['MonthlyIncome'].mean()
                stayed_avg_income = df_clean[df_clean['Attrition'] == 'No']['MonthlyIncome'].mean()
                income_diff = stayed_avg_income - left_avg_income
                
                st.markdown(f"""
                <div style="background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6; margin: 8px 0; border-left: 4px solid #28a745;">
                    <h5 style="color: #155724; margin: 0 0 8px 0;">Income Analysis</h5>
                    <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Active Emp. Avg:</strong> ${stayed_avg_income:,.0f}</p>
                    <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Left Emp. Avg:</strong> ${left_avg_income:,.0f}</p>
                    <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Difference:</strong> ${income_diff:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Age analysis
            young_attrition = df_clean[df_clean['Age'] < 30]['Attrition'].apply(lambda x: x == 'Yes').mean() * 100
            old_attrition = df_clean[df_clean['Age'] > 50]['Attrition'].apply(lambda x: x == 'Yes').mean() * 100
            
            st.markdown(f"""
            <div style="background: #ffffff; padding: 18px; border-radius: 8px; border: 1px solid #dee2e6; margin: 8px 0; border-left: 4px solid #6f42c1;">
                <h5 style="color: #6f42c1; margin: 0 0 8px 0;">Age Demographics</h5>
                <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Under 30 Turnover:</strong> {young_attrition:.1f}%</p>
                <p style="margin: 4px 0; color: #6c757d; font-size: 0.9em;"><strong>Over 50 Turnover:</strong> {old_attrition:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

def department_breakdown_analysis(df):
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="enterprise-header">
        <h1>Department Breakdown Analysis</h1>
        <p>Comprehensive departmental performance and workforce distribution</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Department' not in df_clean.columns:
        st.error("Department data is not available in the dataset")
        return
    
    departments = df_clean['Department'].unique()
    
    for i, dept in enumerate(departments):
        with st.expander(f"{dept} Department Analysis", expanded=(i==0)):
            dept_data = df_clean[df_clean['Department'] == dept]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Staff", len(dept_data))
            with col2:
                dept_attrition = (dept_data['Attrition'] == 'Yes').mean() * 100
                st.metric("Turnover Rate", f"{dept_attrition:.1f}%")
            with col3:
                avg_age_dept = dept_data['Age'].mean()
                st.metric("Avg Age", f"{avg_age_dept:.1f}")
            with col4:
                avg_tenure_dept = dept_data['YearsAtCompany'].mean()
                st.metric("Avg Tenure", f"{avg_tenure_dept:.1f}y")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                attrition_count = dept_data['Attrition'].value_counts()
                fig = px.bar(
                    x=attrition_count.index,
                    y=attrition_count.values,
                    title=f"{dept} - Employee Status",
                    color=attrition_count.index,
                    color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
                )
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, width='stretch')
            
            with col_right:
                if 'JobRole' in dept_data.columns:
                    role_count = dept_data['JobRole'].value_counts().head(5)
                    fig = px.pie(
                        values=role_count.values,
                        names=role_count.index,
                        title=f"{dept} - Job Roles"
                    )
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, width='stretch')
    
    st.markdown("### Department Comparison Overview")
    
    dept_summary = []
    for dept in departments:
        dept_data = df_clean[df_clean['Department'] == dept]
        dept_summary.append({
            'Department': dept,
            'Total Employees': len(dept_data),
            'Turnover Rate (%)': (dept_data['Attrition'] == 'Yes').mean() * 100,
            'Avg Age': dept_data['Age'].mean(),
            'Avg Tenure': dept_data['YearsAtCompany'].mean()
        })
    
    dept_df = pd.DataFrame(dept_summary)
    
    st.dataframe(
        dept_df.style.background_gradient(subset=['Turnover Rate (%)']),
        width='stretch'
    )
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="enterprise-header">
        <h1>Employee Dataset Explorer</h1>
        <p>Comprehensive data analysis and statistical insights</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-divider">Dataset Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        show_data = st.checkbox("Show Complete Dataset")
        if show_data:
            st.subheader("Employee Dataset")
            st.dataframe(df_clean, width='stretch', height=400)
    
    with col2:
        st.markdown(f"""
        <div class="data-panel">
            <h4>Dataset Summary</h4>
            <p><strong>Total Records:</strong> {len(df_clean):,}</p>
            <p><strong>Features:</strong> {len(df_clean.columns)}</p>
            <p><strong>Data Quality:</strong> Good</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_data = df_clean.isnull().sum().sum()
        completeness = ((len(df_clean) * len(df_clean.columns) - missing_data) / (len(df_clean) * len(df_clean.columns))) * 100
        st.markdown(f"""
        <div class="insights-panel">
            <h4>Data Completeness</h4>
            <p><strong>Complete:</strong> {completeness:.1f}%</p>
            <p><strong>Missing:</strong> {missing_data}</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature analysis
    st.markdown('<div class="content-divider">Feature Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_column = st.selectbox("Select Feature for Analysis", df_clean.columns)
    
    with col2:
        analysis_type = st.radio("Analysis Type", ["Distribution", "Statistics"], horizontal=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
        st.subheader(f"{analysis_type}: {selected_column}")
        
        if analysis_type == "Distribution":
            if df_clean[selected_column].dtype in ['object', 'category']:
                fig = px.histogram(
                    df_clean, 
                    x=selected_column, 
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#3498db']
                )
            else:
                fig = px.histogram(
                    df_clean, 
                    x=selected_column, 
                    nbins=30,
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#3498db']
                )
        else:
            if df_clean[selected_column].dtype in ['int64', 'float64']:
                fig = px.box(
                    df_clean, 
                    y=selected_column, 
                    title=f"Statistics of {selected_column}",
                    color_discrete_sequence=['#3498db']
                )
            else:
                value_counts = df_clean[selected_column].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6']
                )
        
        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader(f"Insights for {selected_column}")
        
        if df_clean[selected_column].dtype in ['int64', 'float64']:
            stats = df_clean[selected_column].describe()
            st.markdown(f"""
            <div class="data-panel">
                <h4>Statistical Summary</h4>
                <p><strong>Mean:</strong> {stats['mean']:.2f}</p>
                <p><strong>Median:</strong> {stats['50%']:.2f}</p>
                <p><strong>Std Dev:</strong> {stats['std']:.2f}</p>
                <p><strong>Min:</strong> {stats['min']:.2f}</p>
                <p><strong>Max:</strong> {stats['max']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            value_counts = df_clean[selected_column].value_counts().head(10)
            st.markdown('<div class="data-panel"><h4>Top Categories</h4></div>', unsafe_allow_html=True)
            
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                percentage = (count / len(df_clean)) * 100
                st.markdown(f"**{idx}.** {value}: **{count:,}** ({percentage:.1f}%)")

def statistical_analysis_module(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Principal Component Analysis</h1>
        <p>Advanced statistical dimensionality reduction and pattern discovery</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="alert-panel">
            <h4>Insufficient Data</h4>
            <p>PCA requires at least 2 numeric features.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.markdown('<div class="section-box">PCA Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        max_components = min(10, len(numeric_cols))
        n_components = st.slider("Number of Components", 2, max_components, min(3, max_components))
    
    with col2:
        st.markdown(f"""
        <div class="data-panel">
            <h4>Analysis Parameters</h4>
            <p><strong>Features:</strong> {len(numeric_cols)}</p>
            <p><strong>Components:</strong> {n_components}</p>
            <p><strong>Samples:</strong> {len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    st.markdown('<div class="section-box">Results & Visualization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
        st.subheader("Variance Explained")
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(1, n_components + 1)),
            y=pca.explained_variance_ratio_,
            name='Individual Variance',
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(1, n_components + 1)),
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='#e74c3c', width=3)
        ))
        
        fig.update_layout(
            title="Variance Explained by Components",
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
        st.subheader("Component Visualization")
        
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=df['Attrition'] if 'Attrition' in df.columns else None,
            title="Data Projection on First Two Components",
            labels={
                'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'
            },
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        
        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

def correlation_analysis_module(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Feature Correlation Study</h1>
        <p>Advanced feature relationships and importance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="alert-panel">
            <h4>Limited Features</h4>
            <p>Analysis requires more numeric features.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<div class="section-box">Feature Correlation Analysis</div>', unsafe_allow_html=True)
    
    corr_matrix = df[numeric_cols].corr()
    
    st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    fig.update_layout(height=600, font=dict(size=12))
    st.plotly_chart(fig, width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

def ml_performance_analysis(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Machine Learning Model Analysis</h1>
        <p>Advanced predictive model evaluation and comparison framework</p>
    </div>
    """, unsafe_allow_html=True)

    if 'Attrition' not in df.columns:
        st.markdown("""
        <div class="alert-panel">
            <h4>Missing Target Variable</h4>
            <p>Attrition column is required for model analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Data preprocessing
    df_processed = preprocess_data(df)
    
    if 'Attrition' in df_processed.columns:
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train models and collect results
        results = {}
        feature_importance_rf = None
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate ROC-AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            except:
                roc_auc = 0
            
            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC': roc_auc
            }
            
            # Store feature importance from Random Forest
            if name == 'Random Forest':
                feature_importance_rf = model.feature_importances_
        
        # Model Comparison Section
        st.markdown('<div class="section-box">Model Comparison Results</div>', unsafe_allow_html=True)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(3)
        
        # Display comparison table
        st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
        st.subheader("Performance Metrics Comparison")
        st.dataframe(comparison_df, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualize model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
            fig = px.bar(
                comparison_df.reset_index(),
                x='index',
                y=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                title="Model Performance Comparison",
                color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#27ae60']
            )
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
            fig = px.bar(
                comparison_df.reset_index(),
                x='index',
                y='ROC-AUC',
                title="ROC-AUC Score Comparison",
                color_discrete_sequence=['#9b59b6']
            )
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance Analysis
        if feature_importance_rf is not None:
            st.markdown('<div class="section-box">Feature Importance Analysis</div>', unsafe_allow_html=True)
            
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance_rf
            }).sort_values('Importance', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="visualization-frame">', unsafe_allow_html=True)
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(height=500, font=dict(size=12))
                st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="data-panel">', unsafe_allow_html=True)
                st.markdown("### Key Attrition Factors")
                for i, row in importance_df.head(5).iterrows():
                    percentage = row['Importance'] * 100
                    st.markdown(f"**{row['Feature']}**: {percentage:.1f}% importance")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # HR Insights and Recommendations
        st.markdown('<div class="section-box">HR Insights & Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insights-panel">
                <h4>Key Findings</h4>
                <p><strong>Best Model:</strong> """ + max(results.keys(), key=lambda k: results[k]['ROC-AUC']) + """</p>
                <p><strong>Highest Accuracy:</strong> """ + f"{max(results.values(), key=lambda x: x['Accuracy'])['Accuracy']:.1%}" + """</p>
                <p><strong>Risk Prediction:</strong> Model can identify high-risk employees with """ + f"{max(results.values(), key=lambda x: x['ROC-AUC'])['ROC-AUC']:.1%}" + """ accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="data-panel">
                <h4>Recommended HR Strategies</h4>
                <p><strong>1. Focus on High-Impact Areas:</strong> Target the top 3 identified factors</p>
                <p><strong>2. Early Intervention:</strong> Use model to identify at-risk employees early</p>
                <p><strong>3. Regular Monitoring:</strong> Track key metrics monthly</p>
                <p><strong>4. Targeted Retention:</strong> Develop specific programs for high-risk groups</p>
                <p><strong>5. Exit Interviews:</strong> Validate model predictions with departing employees</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Export Feature
        st.markdown('<div class="section-box">Model Export & Deployment</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get best performing model
            best_model_name = max(results.keys(), key=lambda k: results[k]['ROC-AUC'])
            best_model = models[best_model_name]
            
            st.markdown(f"""
            <div class="data-panel">
                <h4>Best Model: {best_model_name}</h4>
                <p><strong>ROC-AUC Score:</strong> {results[best_model_name]['ROC-AUC']:.3f}</p>
                <p><strong>Accuracy:</strong> {results[best_model_name]['Accuracy']:.3f}</p>
                <p><strong>Status:</strong> Ready for deployment</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Save model button
            if st.button("Export Best Model", type="primary"):
                try:
                    # Save the best model
                    model_filename = f"best_attrition_model_{best_model_name.lower().replace(' ', '_')}.pkl"
                    joblib.dump(best_model, model_filename)
                    
                    # Save preprocessing info
                    preprocessing_info = {
                        'feature_columns': list(X.columns),
                        'model_type': best_model_name,
                        'performance_metrics': results[best_model_name]
                    }
                    joblib.dump(preprocessing_info, f"model_info_{best_model_name.lower().replace(' ', '_')}.pkl")

                    st.success(f"Model exported successfully!")
                    st.info(f"Saved as: {model_filename}")
                    st.info(f"Model info saved as: model_info_{best_model_name.lower().replace(' ', '_')}.pkl")
                    
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        
        with col2:
            st.markdown("""
            <div class="insights-panel">
                <h4>Deployment Instructions</h4>
                <p><strong>1. Model Files:</strong> Download the .pkl files generated</p>
                <p><strong>2. Load Model:</strong> Use joblib.load() in production</p>
                <p><strong>3. Preprocessing:</strong> Apply same transformations as training</p>
                <p><strong>4. Prediction:</strong> Use model.predict_proba() for risk scores</p>
                <p><strong>5. Integration:</strong> Embed in HR systems or dashboards</p>
            </div>
            """, unsafe_allow_html=True)

def employee_risk_prediction(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Employee Attrition Risk Assessment</h1>
        <p>Individual employee turnover risk evaluation system</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-divider">Risk Assessment Interface</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=500000, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
    
    with col2:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        overtime = st.selectbox("OverTime", ["Yes", "No"])
    
    if st.button("Predict Attrition Risk"):
        risk_score = 0
        
        # Age factor
        if age < 25 or age > 55:
            risk_score += 0.2
        
        # Income factor
        if monthly_income < 3000:
            risk_score += 0.3
        
        # Satisfaction factor
        if job_satisfaction <= 2:
            risk_score += 0.3
        
        # Work-life balance factor
        if work_life_balance <= 2:
            risk_score += 0.2
        
        # Overtime factor
        if overtime == "Yes":
            risk_score += 0.2
        
        risk_percentage = min(risk_score * 100, 95)
        
        if risk_percentage < 30:
            color = "#27ae60"
            risk_level = "Low"
        elif risk_percentage < 60:
            color = "#f39c12"
            risk_level = "Medium"
        else:
            color = "#e74c3c"
            risk_level = "High"
        
        st.markdown(f"""
        <div class="analytics-card" style="margin-top: 20px;">
            <h2 style="color: {color}; margin: 0;">{risk_percentage:.1f}%</h2>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Turnover Risk Level ({risk_level})</p>
        </div>
        """, unsafe_allow_html=True)

def employee_records_viewer(df):
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="enterprise-header">
        <h1>Employee Records Viewer</h1>
        <p>Browse and filter employee database records</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Filter & Search Options")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        status_filter = st.multiselect(
            "Employment Status",
            options=['Yes', 'No'],
            default=['Yes', 'No']
        )
    
    with filter_col2:
        if 'Department' in df_clean.columns:
            dept_filter = st.multiselect(
                "Department",
                options=df_clean['Department'].unique(),
                default=df_clean['Department'].unique()
            )
        else:
            dept_filter = []
    
    with filter_col3:
        age_range = st.slider(
            "Age Range",
            min_value=int(df_clean['Age'].min()),
            max_value=int(df_clean['Age'].max()),
            value=(int(df_clean['Age'].min()), int(df_clean['Age'].max()))
        )
    
    with filter_col4:
        show_count = st.number_input(
            "Records to Display",
            min_value=10,
            max_value=len(df_clean),
            value=100
        )
    
    # Apply filters
    filtered_df = df_clean[
        (df_clean['Attrition'].isin(status_filter)) &
        (df_clean['Age'] >= age_range[0]) &
        (df_clean['Age'] <= age_range[1])
    ]
    
    if dept_filter and 'Department' in df_clean.columns:
        filtered_df = filtered_df[filtered_df['Department'].isin(dept_filter)]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Showing:** {min(show_count, len(filtered_df))} of {len(filtered_df)} records")
    with col2:
        st.info(f"**Active Employees:** {len(filtered_df[filtered_df['Attrition'] == 'No'])}")
    with col3:
        st.info(f"**Former Employees:** {len(filtered_df[filtered_df['Attrition'] == 'Yes'])}")
    
    st.markdown("### Employee Data Table")
    
    available_columns = df_clean.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:",
        options=available_columns,
        default=['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'Attrition'] if all(col in available_columns for col in ['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'Attrition']) else available_columns[:6]
    )
    
    if selected_columns:
        display_df = filtered_df[selected_columns].head(show_count)
        
        if 'Attrition' in selected_columns:
            def color_attrition(val):
                if val == 'Yes':
                    return 'background-color: #ffebee; color: #d32f2f; font-weight: bold;'
                else:
                    return 'background-color: #e8f5e8; color: #388e3c; font-weight: bold;'
            
            styled_df = display_df.style.map(color_attrition, subset=['Attrition'])
            st.dataframe(styled_df, width='stretch', height=400)
        else:
            st.dataframe(display_df, width='stretch', height=400)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"employee_records_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def demographic_insights_analysis(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Demographic Insights Analysis</h1>
        <p>Age, gender, and demographic patterns across the workforce</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Age Analysis", "Experience Levels", "Income Distribution"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(
                df, 
                x='Age', 
                color='Attrition',
                nbins=20,
                title='Age Distribution by Employment Status',
                color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'},
                barmode='overlay'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Age group analysis
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['Under 30', '30-39', '40-49', '50+'])
            age_attrition = df.groupby('AgeGroup', observed=True)['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
            
            st.markdown("### Turnover by Age Group")
            for age_group, rate in age_attrition.items():
                st.metric(f"{age_group}", f"{rate:.1f}%")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                df,
                x='YearsAtCompany',
                y='Age',
                color='Attrition',
                title='Experience vs Age',
                color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.box(
                df,
                x='Attrition',
                y='YearsAtCompany',
                title='Tenure Distribution by Status',
                color='Attrition',
                color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
            )
            st.plotly_chart(fig, width='stretch')
    
    with tab3:
        if 'MonthlyIncome' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df,
                    x='MonthlyIncome',
                    color='Attrition',
                    title='Income Distribution',
                    nbins=20,
                    color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                income_stats = df.groupby('Attrition', observed=True)['MonthlyIncome'].agg(['mean', 'median'])
                st.markdown("### Income Statistics")
                st.dataframe(income_stats)

def pattern_discovery_analysis(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Pattern Discovery Analysis</h1>
        <p>Statistical patterns and correlations in employee data</p>
    </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Insufficient numeric columns for pattern analysis")
        return
    
    section = st.radio("Select Analysis Type", ["Correlation Matrix", "Principal Components", "Statistical Relationships"])
    
    if section == "Correlation Matrix":
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("### Strong Correlations (>0.5)")
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': f"{corr_val:.3f}"
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr))
        else:
            st.info("No strong correlations found")
    
    elif section == "Principal Components":
        # PCA analysis with different visualization
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_components = st.slider("Number of Components", 2, min(10, len(numeric_cols)), 3)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            # Variance explained
            fig = px.bar(
                x=range(1, n_components + 1),
                y=pca.explained_variance_ratio_,
                title="Variance Explained by Components"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # 2D scatter plot
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                color=df['Attrition'],
                title="First Two Principal Components",
                color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
            )
            st.plotly_chart(fig, width='stretch')
    
    elif section == "Statistical Relationships":
        # Statistical relationships analysis
        st.markdown("### Feature Relationship Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis Feature", numeric_cols)
        with col2:
            y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        if x_feature != y_feature:
            # Calculate correlation
            correlation = df[x_feature].corr(df[y_feature])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                try:
                    fig = px.scatter(
                        df,
                        x=x_feature,
                        y=y_feature,
                        color='Attrition' if 'Attrition' in df.columns else None,
                        title=f"Relationship: {x_feature} vs {y_feature}",
                        trendline="ols",
                        color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
                    )
                except Exception as e:
                    st.warning("Trendline analysis requires additional packages. Showing basic scatter plot.")
                    fig = px.scatter(
                        df,
                        x=x_feature,
                        y=y_feature,
                        color='Attrition' if 'Attrition' in df.columns else None,
                        title=f"Relationship: {x_feature} vs {y_feature}",
                        color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
                    )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown(f"""
                <div style="background: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin: 10px 0;">
                    <h4 style="color: #2c3e50; margin: 0 0 15px 0;">Statistical Summary</h4>
                    <p style="margin: 5px 0; color: #34495e;"><strong>Correlation:</strong> {correlation:.3f}</p>
                    <p style="margin: 5px 0; color: #34495e;"><strong>Relationship:</strong> {"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"}</p>
                    <p style="margin: 5px 0; color: #34495e;"><strong>Direction:</strong> {"Positive" if correlation > 0 else "Negative"}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature statistics
                x_stats = df[x_feature].describe()
                y_stats = df[y_feature].describe()
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;">
                    <h5 style="color: #495057; margin: 0 0 10px 0;">{x_feature} Stats</h5>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Mean: {x_stats['mean']:.2f}</p>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Std: {x_stats['std']:.2f}</p>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Range: {x_stats['min']:.1f} - {x_stats['max']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0;">
                    <h5 style="color: #495057; margin: 0 0 10px 0;">{y_feature} Stats</h5>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Mean: {y_stats['mean']:.2f}</p>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Std: {y_stats['std']:.2f}</p>
                    <p style="margin: 3px 0; color: #6c757d; font-size: 0.9em;">Range: {y_stats['min']:.1f} - {y_stats['max']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### Advanced Statistical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feature Distributions")
            selected_feature = st.selectbox("Select Feature for Distribution", numeric_cols)
            
            if 'Attrition' in df.columns:
                fig = px.histogram(
                    df,
                    x=selected_feature,
                    color='Attrition',
                    title=f"Distribution of {selected_feature} by Attrition",
                    nbins=20,
                    color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'},
                    barmode='overlay',
                    opacity=0.7
                )
            else:
                fig = px.histogram(
                    df,
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}",
                    nbins=20,
                    color_discrete_sequence=['#3498db']
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Statistical Comparison")
            box_feature = st.selectbox("Select Feature for Box Plot", numeric_cols, key="box_feature")
            
            if 'Attrition' in df.columns:
                fig = px.box(
                    df,
                    x='Attrition',
                    y=box_feature,
                    title=f"{box_feature} by Employment Status",
                    color='Attrition',
                    color_discrete_map={'No': '#27ae60', 'Yes': '#e74c3c'}
                )
            else:
                fig = px.box(
                    df,
                    y=box_feature,
                    title=f"{box_feature} Distribution",
                    color_discrete_sequence=['#3498db']
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Statistical tests and insights
        if 'Attrition' in df.columns:
            st.markdown("### Key Statistical Insights")
            
            attrition_stats = df.groupby('Attrition', observed=True)[numeric_cols].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Average Values by Employment Status")
                st.dataframe(attrition_stats.round(2), width='stretch')
            
            with col2:
                st.markdown("#### Notable Differences")
                differences = []
                for col in numeric_cols:
                    if len(df[df['Attrition'] == 'Yes']) > 0 and len(df[df['Attrition'] == 'No']) > 0:
                        yes_mean = df[df['Attrition'] == 'Yes'][col].mean()
                        no_mean = df[df['Attrition'] == 'No'][col].mean()
                        diff_pct = ((yes_mean - no_mean) / no_mean * 100) if no_mean != 0 else 0
                        
                        if abs(diff_pct) > 10: 
                            differences.append({
                                'Feature': col,
                                'Difference': f"{diff_pct:+.1f}%",
                                'Direction': '↑ Higher' if diff_pct > 0 else '↓ Lower'
                            })
                
                if differences:
                    diff_df = pd.DataFrame(differences)
                    st.dataframe(diff_df, width='stretch')
                else:
                    st.info("No significant differences found (>10%)")
        else:
            st.info("Attrition data not available for group comparisons")

def model_evaluation_report(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Model Evaluation Report</h1>
        <p>Comprehensive machine learning model assessment and comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Attrition' not in df.columns:
        st.error("Attrition column required for model evaluation")
        return
    
    df_processed = preprocess_data(df)
    
    if 'Attrition' in df_processed.columns:
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = []
        feature_importance_rf = None
        
        progress_bar = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            except:
                roc_auc = 0
            
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy:.3f}",
                'Precision': f"{precision:.3f}",
                'Recall': f"{recall:.3f}",
                'F1-Score': f"{f1:.3f}",
                'ROC-AUC': f"{roc_auc:.3f}"
            })
            
            if name == 'Random Forest':
                feature_importance_rf = model.feature_importances_
            
            progress_bar.progress((i + 1) / len(models))
        
        st.markdown("### Model Performance Comparison")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width='stretch')
        
        if feature_importance_rf is not None:
            st.markdown("### Feature Importance (Random Forest)")
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance_rf
            }).sort_values('Importance', ascending=True).tail(10)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Important Features"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')

def individual_risk_calculator(df):
    st.markdown("""
    <div class="enterprise-header">
        <h1>Individual Risk Calculator</h1>
        <p>Calculate turnover risk for individual employees</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Employee Information Input")
    
    with st.form("risk_assessment_form"):
        st.markdown("**Step 1: Basic Information**")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Employee Age", min_value=18, max_value=70, value=30)
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=500000, value=5000)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        
        with col2:
            job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
            work_life_balance = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=3)
            overtime = st.selectbox("Works Overtime", ["No", "Yes"])
        
        st.markdown("**Step 2: Calculate Risk**")
        submitted = st.form_submit_button("Calculate Turnover Risk", width='stretch')
        
        if submitted:
            risk_factors = []
            risk_score = 0
            
            # Age factor
            if age < 25:
                risk_score += 0.25
                risk_factors.append("Young employee (higher mobility)")
            elif age > 55:
                risk_score += 0.15
                risk_factors.append("Nearing retirement age")
            
            # Income factor
            if monthly_income < 3000:
                risk_score += 0.3
                risk_factors.append("Below average compensation")
            elif monthly_income > 8000:
                risk_score -= 0.1
                risk_factors.append("Above average compensation")
            
            # Satisfaction factors
            if job_satisfaction <= 2:
                risk_score += 0.35
                risk_factors.append("Low job satisfaction")
            if work_life_balance <= 2:
                risk_score += 0.25
                risk_factors.append("Poor work-life balance")
            
            # Tenure factor
            if years_at_company < 2:
                risk_score += 0.2
                risk_factors.append("New employee (settling period)")
            elif years_at_company > 10:
                risk_score -= 0.15
                risk_factors.append("Long-term employee")
            
            # Overtime factor
            if overtime == "Yes":
                risk_score += 0.15
                risk_factors.append("Regular overtime work")
            
            risk_percentage = min(max(risk_score * 100, 5), 95)  
            
            # Risk level classification
            if risk_percentage < 25:
                risk_level = "Low"
                risk_color = "#27ae60"
            elif risk_percentage < 50:
                risk_level = "Moderate"
                risk_color = "#f39c12"
            elif risk_percentage < 75:
                risk_level = "High"
                risk_color = "#e67e22"
            else:
                risk_level = "Critical"
                risk_color = "#e74c3c"
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: {risk_color}; color: white; padding: 30px; border-radius: 15px; text-align: center;">
                    <h1 style="margin: 0; font-size: 3em;">{risk_percentage:.0f}%</h1>
                    <h3 style="margin: 10px 0 0 0; opacity: 0.9;">{risk_level} Risk</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Risk Factors Identified:")
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("• No significant risk factors identified")
                
                st.markdown("### Recommendations:")
                if risk_percentage >= 75:
                    st.markdown("• Immediate intervention required")
                    st.markdown("• Schedule retention discussion")
                    st.markdown("• Review compensation and role")
                elif risk_percentage >= 50:
                    st.markdown("• Monitor closely")
                    st.markdown("• Address identified concerns")
                    st.markdown("• Consider career development")
                else:
                    st.markdown("• Continue regular engagement")
                    st.markdown("• Maintain current practices")

# Main application
def main():
    # Load data
    try:
        df = load_data()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    st.sidebar.markdown("""
    <div class="content-divider">
        <h3>Analysis Sections</h3>
    </div>
    """, unsafe_allow_html=True)
    
    main_section = st.sidebar.radio(
        "Select Analysis Section",
        ["Company Overview", "Employee Analysis", "Advanced Analytics", "Prediction Tools"]
    )
    
    if main_section == "Company Overview":
        st.sidebar.markdown("**Available Reports:**")
        sub_option = st.sidebar.radio("Company Reports", ["Executive Summary", "Department Analysis"], label_visibility="collapsed")
        if sub_option == "Executive Summary":
            executive_summary_report(df)
        else:
            department_breakdown_analysis(df)
    
    elif main_section == "Employee Analysis":
        st.sidebar.markdown("**Available Reports:**")
        sub_option = st.sidebar.radio("Employee Reports", ["Employee Records", "Demographic Analysis"], label_visibility="collapsed")
        if sub_option == "Employee Records":
            employee_records_viewer(df)
        else:
            demographic_insights_analysis(df)
    
    elif main_section == "Advanced Analytics":
        st.sidebar.markdown("**Available Reports:**")
        sub_option = st.sidebar.radio("Analytics Reports", ["Pattern Analysis", "Model Performance"], label_visibility="collapsed")
        if sub_option == "Pattern Analysis":
            pattern_discovery_analysis(df)
        else:
            model_evaluation_report(df)
    
    elif main_section == "Prediction Tools":
        individual_risk_calculator(df)

if __name__ == "__main__":

    main()

