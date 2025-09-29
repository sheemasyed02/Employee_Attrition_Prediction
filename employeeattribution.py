import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="HR Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, professional CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    
    .header-box {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 30px;
        text-align: center;
    }
    
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .section-box {
        background-color: #ecf0f1;
        padding: 10px;
        border-left: 4px solid #3498db;
        margin: 20px 0 10px 0;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .info-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
        color: #856404;
    }
    
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 10px 0;
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
    
    # Clean Attrition column
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

def show_overview_page(df):
    # Clean data
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    # Header
    st.markdown("""
    <div class="header-box">
        <h1>Human Resources Analytics Dashboard</h1>
        <p>Employee data analysis and workforce insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    st.markdown('<div class="section-box">Key Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(df_clean)
    attrition_rate = (df_clean['Attrition'] == 'Yes').mean() * 100
    avg_tenure = df_clean['YearsAtCompany'].mean()
    avg_age = df_clean['Age'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: #3498db; margin: 0;">{total_employees:,}</h3>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Total Employees</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#e74c3c" if attrition_rate > 15 else "#27ae60"
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: {color}; margin: 0;">{attrition_rate:.1f}%</h3>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Attrition Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: #3498db; margin: 0;">{avg_tenure:.1f}</h3>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Avg Tenure (Years)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3 style="color: #3498db; margin: 0;">{avg_age:.1f}</h3>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Average Age</p>
        </div>
        """, unsafe_allow_html=True)

    # Charts
    st.markdown('<div class="section-box">Workforce Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Employee Retention Status")
        attrition_counts = df_clean['Attrition'].value_counts()
        
        labels = ['Retained Employees' if x == 'No' else 'Left Company' for x in attrition_counts.index]
        colors = ['#3498db', '#e74c3c']
        
        fig = px.pie(
            values=attrition_counts.values, 
            names=labels,
            title='Employee Retention Overview',
            color_discrete_sequence=colors
        )
        
        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Department Distribution")
        
        if 'Department' in df_clean.columns:
            dept_attrition = df_clean.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
            
            fig = px.bar(
                dept_attrition, 
                barmode='group', 
                title='Employees by Department',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            
            fig.update_layout(height=400, font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Department data not available")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Age Distribution")
        
        fig = px.histogram(
            df_clean, 
            x='Age', 
            nbins=20,
            title='Employee Age Distribution',
            color_discrete_sequence=['#3498db']
        )
        
        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Tenure Analysis")
        
        fig = px.box(
            df_clean, 
            y='YearsAtCompany', 
            title='Years at Company Distribution',
            color_discrete_sequence=['#3498db']
        )
        
        fig.update_layout(height=400, font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_data_exploration(df):
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="header-box">
        <h1>Data Exploration & Analysis</h1>
        <p>Detailed data exploration and statistical insights</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-box">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        show_data = st.checkbox("Show Complete Dataset")
        if show_data:
            st.subheader("Employee Dataset")
            st.dataframe(df_clean, use_container_width=True, height=400)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
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
        <div class="success-box">
            <h4>Data Completeness</h4>
            <p><strong>Complete:</strong> {completeness:.1f}%</p>
            <p><strong>Missing:</strong> {missing_data}</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature analysis
    st.markdown('<div class="section-box">Feature Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_column = st.selectbox("Select Feature for Analysis", df_clean.columns)
    
    with col2:
        analysis_type = st.radio("Analysis Type", ["Distribution", "Statistics"], horizontal=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader(f"Insights for {selected_column}")
        
        if df_clean[selected_column].dtype in ['int64', 'float64']:
            stats = df_clean[selected_column].describe()
            st.markdown(f"""
            <div class="info-box">
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
            st.markdown('<div class="info-box"><h4>Top Categories</h4></div>', unsafe_allow_html=True)
            
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                percentage = (count / len(df_clean)) * 100
                st.markdown(f"**{idx}.** {value}: **{count:,}** ({percentage:.1f}%)")

def display_pca_analysis(df):
    st.markdown("""
    <div class="header-box">
        <h1>Principal Component Analysis</h1>
        <p>Dimensionality reduction and pattern analysis</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="warning-box">
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
        <div class="info-box">
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
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_feature_analysis_page(df):
    st.markdown("""
    <div class="header-box">
        <h1>Feature Analysis & Insights</h1>
        <p>Feature relationships and importance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="warning-box">
            <h4>Limited Features</h4>
            <p>Analysis requires more numeric features.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown('<div class="section-box">Feature Correlation Analysis</div>', unsafe_allow_html=True)
    
    corr_matrix = df[numeric_cols].corr()
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    fig.update_layout(height=600, font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_performance_page(df):
    st.markdown("""
    <div class="header-box">
        <h1>Model Performance Analysis</h1>
        <p>Machine learning model evaluation and metrics</p>
    </div>
    """, unsafe_allow_html=True)

    if 'Attrition' not in df.columns:
        st.markdown("""
        <div class="warning-box">
            <h4>Missing Target Variable</h4>
            <p>Attrition column is required for model analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Quick model training for demonstration
    df_processed = preprocess_data(df)
    
    if 'Attrition' in df_processed.columns:
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        st.markdown('<div class="section-box">Model Performance Metrics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #3498db; margin: 0;">{accuracy:.3f}</h3>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #3498db; margin: 0;">{precision:.3f}</h3>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">Precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #3498db; margin: 0;">{recall:.3f}</h3>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">Recall</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #3498db; margin: 0;">{f1:.3f}</h3>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">F1 Score</p>
            </div>
            """, unsafe_allow_html=True)

def prediction_interface_page(df):
    st.markdown("""
    <div class="header-box">
        <h1>Employee Attrition Prediction</h1>
        <p>Predict employee attrition based on individual characteristics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-box">Prediction Interface</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
    
    with col2:
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        overtime = st.selectbox("OverTime", ["Yes", "No"])
    
    if st.button("Predict Attrition Risk"):
        # Simple prediction logic (placeholder)
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
        <div class="metric-box" style="margin-top: 20px;">
            <h2 style="color: {color}; margin: 0;">{risk_percentage:.1f}%</h2>
            <p style="margin: 5px 0 0 0; color: #7f8c8d;">Attrition Risk ({risk_level})</p>
        </div>
        """, unsafe_allow_html=True)

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
    
    # Sidebar navigation
    st.sidebar.markdown("""
    <div class="section-box">
        <h3>Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Data Exploration", "PCA Analysis", "Feature Analysis", "Model Performance", "Prediction"]
    )
    
    # Page routing
    if page == "Overview":
        show_overview_page(df)
    elif page == "Data Exploration":
        display_data_exploration(df)
    elif page == "PCA Analysis":
        display_pca_analysis(df)
    elif page == "Feature Analysis":
        show_feature_analysis_page(df)
    elif page == "Model Performance":
        show_model_performance_page(df)
    elif page == "Prediction":
        prediction_interface_page(df)

if __name__ == "__main__":
    main()