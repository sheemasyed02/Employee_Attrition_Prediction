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
    page_title="Workforce Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Professional purple theme with soft backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    .main {
        padding: 1.5rem;
        font-family: 'Source Sans Pro', sans-serif;
        background-color: #fafafa;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(145deg, #6b46c1, #8b5cf6);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(107, 70, 193, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0.8rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards with natural shadows */
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.15);
    }
    
    /* Section headers with subtle styling */
    .section-header {
        background: rgba(248, 250, 252, 0.8);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        border-left: 4px solid #8b5cf6;
        margin: 1.8rem 0 1.2rem 0;
        backdrop-filter: blur(10px);
    }
    
    .section-header h3 {
        margin: 0;
        color: #374151;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Information boxes */
    .info-box {
        background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #374151;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #92400e;
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #166534;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: white;
        padding: 1.3rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(139, 92, 246, 0.1);
        box-shadow: 0 2px 12px rgba(139, 92, 246, 0.06);
    }
    
    /* Professional button styling */
    div.stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #6b46c1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.2);
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Form controls */
    .stSelectbox > div > div {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* Chart styling */
    .plotly-chart {
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Source Sans Pro', sans-serif;
        color: #374151;
        font-weight: 600;
    }
    
    p, div, span, li {
        font-family: 'Source Sans Pro', sans-serif;
        color: #4b5563;
    }
    
    /* Subtle animations */
    .metric-card, .section-header, .info-box {
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1.5rem;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-card {
            padding: 1.4rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path relative to the script
    file_path = os.path.join(base_path, 'employeedata.csv')
    print(f"Loading data from: {file_path}")  # Log file path for debugging
    # Load the dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle CSV parsing error by skipping bad lines
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        print("CSV parsing error detected. Attempting to load with error handling...")
        df = pd.read_csv(file_path, on_bad_lines='skip')
    
    return df

def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Clean and normalize the Attrition column to handle case sensitivity
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

class CascadeWrapper:
    def __init__(self, main_model, pre_model):
        self.main_model = main_model
        self.pre_model = pre_model

    def predict(self, X):
        # 1) Generate probabilities from the pre_model
        pre_probs = self.pre_model.predict_proba(X)[:, 1].reshape(-1, 1)
        # 2) Append to X
        X_cascade = np.hstack((X, pre_probs))
        # 3) Predict with main_model
        return self.main_model.predict(X_cascade)

    def predict_proba(self, X):
        # If you want a predict_proba method, do something similar:
        pre_probs = self.pre_model.predict_proba(X)[:, 1].reshape(-1, 1)
        X_cascade = np.hstack((X, pre_probs))
        return self.main_model.predict_proba(X_cascade)


class CatBoostKNNWrapper:
    def __init__(self, main_model, pre_model):
        self.main_model = main_model
        self.pre_model = pre_model

    def predict(self, X):
        # 1) Generate probabilities from the CatBoost model
        catboost_probs = self.pre_model.predict_proba(X)[:, 1].reshape(-1, 1)
        # 2) Append to X
        X_hybrid = np.hstack((X, catboost_probs))
        # 3) Predict with KNN
        return self.main_model.predict(X_hybrid)

    def predict_proba(self, X):
        catboost_probs = self.pre_model.predict_proba(X)[:, 1].reshape(-1, 1)
        X_hybrid = np.hstack((X, catboost_probs))
        return self.main_model.predict_proba(X_hybrid)

# 1. Load pre-trained models
def load_models():
    import traceback  # For detailed exception traceback

    # Dynamically determine the path to the "Models" directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_path, 'Models')  # Adjusted for relative path

    # Check if Models directory exists
    if not os.path.exists(save_path):
        print(f"Models directory not found at: {save_path}")
        print("Creating an empty models dictionary. Please train models first or provide pre-trained models.")
        return {}

    trained_models = {}
    model_names = [
        "Stacked RF+GB+SVM",
        "Cascading Classifiers",
        "Calibration Curves",
        "HGBoost+KNN",
        "XGBRF",
        "CatBoost+KNN",
        "CatBoost",
        "Random_Forest",
        "Bagging"
    ]

    # Load all models
    for model_name in model_names:
        try:
            # Build file path and verify existence
            file_name = os.path.join(save_path, f"{model_name.replace(' ', '_')}.joblib")
            if not os.path.exists(file_name):
                print(f"Model file not found: {file_name}")
                continue
            
            # Debug log for file path
            print(f"Attempting to load {model_name} from {file_name}...")
            
            trained_models[model_name] = joblib.load(file_name)
            print(f"Successfully loaded {model_name}.")

        except Exception as e:
            print(f"Error loading {model_name} from {file_name}: {e}")
            traceback.print_exc()  # Print detailed exception traceback for debugging

    # ----------------------------------------------------------------
    # Wrap "Cascading Classifiers" with the pre-model = "Random_Forest"
    # ----------------------------------------------------------------
    try:
        if "Cascading Classifiers" in trained_models and "Random_Forest" in trained_models:
            main_model = trained_models["Cascading Classifiers"]
            pre_model = trained_models["Random_Forest"]
            cascade_wrapper = CascadeWrapper(main_model=main_model, pre_model=pre_model)
            trained_models["Cascading Classifiers"] = cascade_wrapper
            # Remove "Random_Forest" since it's only a pre-model
            del trained_models["Random_Forest"]
            print("Wrapped 'Cascading Classifiers' with 'Random_Forest'.")
    except Exception as e:
        print(f"Error wrapping 'Cascading Classifiers' with 'Random_Forest': {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------
    # Wrap "CatBoost+KNN" with the pre-model = "CatBoost"
    # ----------------------------------------------------------------
    try:
        if "CatBoost+KNN" in trained_models and "CatBoost" in trained_models:
            main_model = trained_models["CatBoost+KNN"]  # the KNN
            pre_model = trained_models["CatBoost"]
            
            # Log for diagnostics
            print(f"CatBoost model loaded: {pre_model}")
            print(f"KNN model loaded: {main_model}")
            
            catboost_knn_wrapper = CatBoostKNNWrapper(main_model=main_model, pre_model=pre_model)
            trained_models["CatBoost+KNN"] = catboost_knn_wrapper
            # Remove "CatBoost" since it's only a pre-model
            del trained_models["CatBoost"]
            print("Wrapped 'CatBoost+KNN' with 'CatBoost'.")
    except Exception as e:
        print(f"Error wrapping 'CatBoost+KNN' with 'CatBoost': {e}")
        traceback.print_exc()

    # Debug log for final models loaded
    print("Final loaded models:", list(trained_models.keys()))
    
    return trained_models
    
def show_overview_page(df):
    # Clean and normalize the Attrition column to handle case sensitivity
    df_clean = df.copy()
    df_clean['Attrition'] = df_clean['Attrition'].str.upper()
    df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>Workforce Analytics Dashboard</h1>
        <p>Comprehensive employee data analysis and retention insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics section
    st.markdown('<div class="section-header"><h3>Key Performance Indicators</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(df_clean)
    attrition_rate = (df_clean['Attrition'] == 'Yes').mean() * 100
    avg_tenure = df_clean['YearsAtCompany'].mean()
    avg_age = df_clean['Age'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b46c1; margin: 0; font-weight: 600;">Total Workforce</h4>
            <h2 style="color: #374151; margin: 12px 0; font-weight: 700;">{total_employees:,}</h2>
            <p style="margin: 0; color: #6b7280; font-weight: 400;">Active Employees</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#dc2626" if attrition_rate > 15 else "#059669"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b46c1; margin: 0; font-weight: 600;">Attrition Rate</h4>
            <h2 style="color: {color}; margin: 12px 0; font-weight: 700;">{attrition_rate:.1f}%</h2>
            <p style="margin: 0; color: #6b7280; font-weight: 400;">Annual Turnover</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b46c1; margin: 0; font-weight: 600;">Average Tenure</h4>
            <h2 style="color: #374151; margin: 12px 0; font-weight: 700;">{avg_tenure:.1f}</h2>
            <p style="margin: 0; color: #6b7280; font-weight: 400;">Years at Company</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6b46c1; margin: 0; font-weight: 600;">Average Age</h4>
            <h2 style="color: #374151; margin: 12px 0; font-weight: 700;">{avg_age:.1f}</h2>
            <p style="margin: 0; color: #6b7280; font-weight: 400;">Years Old</p>
        </div>
        """, unsafe_allow_html=True)

    # Charts section
    st.markdown('<div class="section-header"><h3>Workforce Distribution Analysis</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Employee Retention Overview")
        attrition_counts = df_clean['Attrition'].value_counts()
        
        # Create proper labels based on actual data
        labels = []
        colors = []
        for value in attrition_counts.index:
            if value == 'No':
                labels.append('Retained Employees')
                colors.append('#8b5cf6')
            else:  # 'Yes'
                labels.append('Left Company')
                colors.append('#f87171')
        
        fig = px.pie(
            values=attrition_counts.values, 
            names=labels,
            title='Current Retention Status',
            color_discrete_sequence=colors,
            hole=0.3
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=11,
            marker=dict(line=dict(color='white', width=2))
        )
        
        fig.update_layout(
            showlegend=True,
            height=400,
            font=dict(size=12, family="Source Sans Pro"),
            title_font_size=14,
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Department Analysis")
        dept_attrition = df_clean.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        
        fig = px.bar(
            dept_attrition, 
            barmode='group', 
            title='Employee Distribution by Department',
            color_discrete_sequence=['#8b5cf6', '#f87171'],
            labels={'value': 'Number of Employees', 'index': 'Department'}
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Department",
            yaxis_title="Number of Employees",
            font=dict(size=12, family="Source Sans Pro"),
            title_font_size=14,
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                title=""
            )
        )
        
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=1
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Additional insights section
    st.markdown('<div class="section-header"><h3>Additional Workforce Insights</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(
            df_clean, 
            x='Age', 
            nbins=20,
            title='Employee Age Distribution',
            color_discrete_sequence=['#8b5cf6']
        )
        fig.update_layout(
            height=280,
            font=dict(size=11, family="Source Sans Pro"),
            title_font_size=13,
            title_x=0.5,
            xaxis_title="Age",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Income by Status")
        fig = px.box(
            df_clean, 
            x='Attrition', 
            y='MonthlyIncome',
            title='Monthly Income Distribution',
            color='Attrition',
            color_discrete_sequence=['#8b5cf6', '#f87171']
        )
        fig.update_layout(
            height=280,
            font=dict(size=11, family="Source Sans Pro"),
            title_font_size=13,
            title_x=0.5,
            showlegend=False,
            xaxis_title="Employment Status",
            yaxis_title="Monthly Income"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Job Satisfaction")
        job_sat = df_clean['JobSatisfaction'].value_counts().sort_index()
        satisfaction_labels = ['Low', 'Medium', 'High', 'Very High'][:len(job_sat)]
        
        fig = px.bar(
            x=satisfaction_labels,
            y=job_sat.values,
            title='Job Satisfaction Levels',
            color=job_sat.values,
            color_continuous_scale=[[0, '#ddd6fe'], [1, '#6b46c1']]
        )
        fig.update_layout(
            height=280,
            font=dict(size=11, family="Source Sans Pro"),
            title_font_size=13,
            title_x=0.5,
            showlegend=False,
            xaxis_title="Satisfaction Level",
            yaxis_title="Number of Employees"
        )
        st.plotly_chart(fig, use_container_width=True)
    

def display_data_exploration(df):
    # Clean and normalize the Attrition column to handle case sensitivity
    df_clean = df.copy()
    if 'Attrition' in df_clean.columns:
        df_clean['Attrition'] = df_clean['Attrition'].str.upper()
        df_clean['Attrition'] = df_clean['Attrition'].replace({'YES': 'Yes', 'NO': 'No'})
    
    st.markdown("""
    <div class="main-header">
        <h1>Data Exploration & Analysis</h1>
        <p>Deep dive into workforce data patterns and distributions</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset overview section
    st.markdown('<div class="section-header"><h3>Dataset Overview</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        show_data = st.checkbox("Display Complete Dataset", help="Toggle to show/hide the full dataset")
        if show_data:
            st.subheader("Complete Employee Dataset")
            st.dataframe(
                df_clean.style.highlight_max(axis=0), 
                use_container_width=True, 
                height=400
            )
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>Dataset Summary</h4>
            <p><strong>Total Records:</strong> {len(df_clean):,}</p>
            <p><strong>Features:</strong> {len(df_clean.columns)}</p>
            <p><strong>Data Quality:</strong> Excellent</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_data = df_clean.isnull().sum().sum()
        completeness = ((len(df_clean) * len(df_clean.columns) - missing_data) / (len(df_clean) * len(df_clean.columns))) * 100
        st.markdown(f"""
        <div class="success-box">
            <h4>Data Completeness</h4>
            <p><strong>Complete:</strong> {completeness:.1f}%</p>
            <p><strong>Missing Values:</strong> {missing_data}</p>
        </div>
        """, unsafe_allow_html=True)

    # Advanced analysis section
    st.markdown('<div class="section-header"><h3>Advanced Column Analysis</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_column = st.selectbox(
            "Select Feature for Analysis", 
            df_clean.columns,
            help="Choose any column to view its distribution and statistical properties"
        )
    
    with col2:
        analysis_type = st.radio(
            "Analysis Type",
            ["Distribution", "Statistical Summary", "Correlation"],
            horizontal=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{analysis_type}: {selected_column}")
        
        if analysis_type == "Distribution":
            if df_clean[selected_column].dtype in ['object', 'category']:
                fig = px.histogram(
                    df_clean, 
                    x=selected_column, 
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#8b5cf6'],
                    text_auto=True
                )
                fig.update_traces(textfont_size=10)
            else:
                fig = px.histogram(
                    df_clean, 
                    x=selected_column, 
                    nbins=30,
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#8b5cf6'],
                    marginal="box"
                )
        
        elif analysis_type == "Statistical Summary":
            if df_clean[selected_column].dtype in ['int64', 'float64']:
                fig = px.box(
                    df_clean, 
                    y=selected_column, 
                    title=f"Statistical Analysis of {selected_column}",
                    color_discrete_sequence=['#8b5cf6'],
                    points="outliers"
                )
            else:
                value_counts = df_clean[selected_column].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe']
                )
        
        else:  # Correlation
            if df_clean[selected_column].dtype in ['int64', 'float64'] and 'Attrition' in df_clean.columns:
                fig = px.scatter(
                    df_clean,
                    x=selected_column,
                    y='MonthlyIncome' if 'MonthlyIncome' in df_clean.columns else df_clean.select_dtypes(include=['int64', 'float64']).columns[0],
                    color='Attrition',
                    title=f"Correlation: {selected_column} vs Income",
                    color_discrete_sequence=['#8b5cf6', '#f87171'],
                    trendline="ols"
                )
            else:
                fig = px.histogram(
                    df_clean, 
                    x=selected_column, 
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=['#8b5cf6']
                )
        
        fig.update_layout(
            height=400,
            showlegend=True,
            font=dict(size=11, family="Source Sans Pro"),
            title_font_size=13,
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Insights for {selected_column}")
        
        if df_clean[selected_column].dtype in ['int64', 'float64']:
            stats = df_clean[selected_column].describe()
            st.markdown(f"""
            <div class="info-box">
                <h4>Statistical Summary</h4>
                <p><strong>Mean:</strong> {stats['mean']:.2f}</p>
                <p><strong>Median:</strong> {stats['50%']:.2f}</p>
                <p><strong>Standard Deviation:</strong> {stats['std']:.2f}</p>
                <p><strong>Minimum:</strong> {stats['min']:.2f}</p>
                <p><strong>Maximum:</strong> {stats['max']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            Q1 = stats['25%']
            Q3 = stats['75%']
            IQR = Q3 - Q1
            outliers = df_clean[(df_clean[selected_column] < Q1 - 1.5*IQR) | (df_clean[selected_column] > Q3 + 1.5*IQR)]
            
            if len(outliers) > 0:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>Outlier Detection</h4>
                    <p><strong>Outliers Found:</strong> {len(outliers)}</p>
                    <p><strong>Percentage:</strong> {(len(outliers)/len(df_clean)*100):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>No Outliers Detected</h4>
                    <p>Data appears to be well-distributed without significant outliers.</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            value_counts = df_clean[selected_column].value_counts().head(10)
            st.markdown(f"""
            <div class="info-box">
                <h4>Top Categories in {selected_column}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for idx, (value, count) in enumerate(value_counts.items(), 1):
                percentage = (count / len(df_clean)) * 100
                st.markdown(f"**{idx}.** {value}: **{count:,}** ({percentage:.1f}%)")
            
            unique_values = df_clean[selected_column].nunique()
            diversity = unique_values / len(df_clean) * 100
            
            st.markdown(f"""
            <div class="success-box">
                <h4>Diversity Metrics</h4>
                <p><strong>Unique Values:</strong> {unique_values}</p>
                <p><strong>Diversity Index:</strong> {diversity:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature correlation heatmap
    if len(df_clean.select_dtypes(include=['int64', 'float64']).columns) > 1:
        st.markdown('<div class="section-header"><h3>Feature Correlation Analysis</h3></div>', unsafe_allow_html=True)
        
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df_clean[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(
            height=500,
            font=dict(size=11, family="Source Sans Pro"),
            title_font_size=14,
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                        'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Notable Feature Correlations")
                st.dataframe(corr_df.round(3), use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="info-box">
                    <h4>Correlation Guide</h4>
                    <p><strong>Strong:</strong> |r| > 0.7</p>
                    <p><strong>Moderate:</strong> 0.3 < |r| < 0.7</p>
                    <p><strong>Interpretation:</strong></p>
                    <p>• Positive: Variables increase together</p>
                    <p>• Negative: One increases, other decreases</p>
                </div>
                """, unsafe_allow_html=True)

def display_pca_analysis(df):
    st.markdown("""
    <div class="main-header">
        <h1>Principal Component Analysis</h1>
        <p>Dimensionality reduction and feature importance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare data for PCA
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="warning-box">
            <h4>Insufficient Numeric Data</h4>
            <p>PCA requires at least 2 numeric features. Please ensure your dataset contains numeric columns.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA configuration
    st.markdown('<div class="section-header"><h3>PCA Configuration</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        max_components = min(10, len(numeric_cols))
        n_components = st.slider(
            "Number of Principal Components", 
            2, max_components, 
            min(3, max_components),
            help="Select the number of principal components to analyze"
        )
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h4>Analysis Parameters</h4>
            <p><strong>Original Features:</strong> {len(numeric_cols)}</p>
            <p><strong>Components Selected:</strong> {n_components}</p>
            <p><strong>Data Points:</strong> {len(df):,}</p>
        </div>
        """, unsafe_allow_html=True)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # Results visualization
    st.markdown('<div class="section-header"><h3>PCA Results & Visualization</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variance Explained Analysis")
        
        # Create cumulative variance plot
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        fig = go.Figure()
        
        # Individual variance
        fig.add_trace(go.Bar(
            x=list(range(1, n_components + 1)),
            y=pca.explained_variance_ratio_,
            name='Individual Variance',
            marker_color='#2c5aa0',
            opacity=0.7
        ))
        
        # Cumulative variance line
        fig.add_trace(go.Scatter(
            x=list(range(1, n_components + 1)),
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Variance Explained by Principal Components",
            xaxis_title="Principal Component",
            yaxis_title="Proportion of Variance Explained",
            height=400,
            showlegend=True,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Variance summary
        total_variance = cumulative_variance[-1] * 100
        st.markdown(f"""
        <div class="success-box">
            <h4>Variance Summary</h4>
            <p><strong>Total Variance Captured:</strong> {total_variance:.1f}%</p>
            <p><strong>First Component:</strong> {pca.explained_variance_ratio_[0]*100:.1f}%</p>
            <p><strong>Second Component:</strong> {pca.explained_variance_ratio_[1]*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Component Visualization")
        
        # 2D scatter plot of first two components
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=df['Attrition'] if 'Attrition' in df.columns else None,
            title="Data Projection on First Two Principal Components",
            labels={
                'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'
            },
            color_discrete_sequence=['#2c5aa0', '#e74c3c']
        )
        
        fig.update_layout(
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance in components
        if st.checkbox("Show Feature Contributions"):
            st.subheader("Feature Contributions to Components")
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame(
                pca.components_[:2].T,
                columns=['PC1', 'PC2'],
                index=numeric_cols
            )
            
            # Sort by absolute contribution to PC1
            feature_importance['Total_Contribution'] = (
                abs(feature_importance['PC1']) + abs(feature_importance['PC2'])
            )
            feature_importance = feature_importance.sort_values('Total_Contribution', ascending=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=feature_importance.index,
                x=feature_importance['PC1'],
                name='PC1 Contribution',
                orientation='h',
                marker_color='#2c5aa0',
                opacity=0.7
            ))
            
            fig.add_trace(go.Bar(
                y=feature_importance.index,
                x=feature_importance['PC2'],
                name='PC2 Contribution',
                orientation='h',
                marker_color='#e74c3c',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Feature Contributions to Principal Components",
                xaxis_title="Component Loading",
                height=max(400, len(feature_importance) * 25),
                barmode='group',
                font=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis_page(df):
    st.markdown("""
    <div class="main-header">
        <h1>Feature Analysis & Insights</h1>
        <p>Comprehensive analysis of feature relationships and importance</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature correlation analysis
    st.markdown('<div class="section-header"><h3>Feature Correlation Matrix</h3></div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) < 2:
        st.markdown("""
        <div class="warning-box">
            <h4>Limited Numeric Features</h4>
            <p>Correlation analysis requires numeric features. Current dataset has limited numeric columns.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Correlation matrix with improved visualization
    corr_matrix = df[numeric_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation Coefficient")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        width=800,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # High correlation pairs
    st.markdown('<div class="section-header"><h3>Strong Feature Relationships</h3></div>', unsafe_allow_html=True)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Threshold for high correlation
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        corr_df = pd.DataFrame(high_corr_pairs)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Highly Correlated Feature Pairs")
            st.dataframe(corr_df.round(3), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Correlation Guide</h4>
                <p><strong>Strong:</strong> |r| > 0.7</p>
                <p><strong>Moderate:</strong> 0.5 < |r| < 0.7</p>
                <p><strong>Weak:</strong> |r| < 0.5</p>
                <p><strong>Positive:</strong> Features increase together</p>
                <p><strong>Negative:</strong> One increases, other decreases</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>No Strong Correlations Found</h4>
            <p>No feature pairs show correlation above 0.5 threshold.</p>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance analysis
    st.markdown('<div class="section-header"><h3>Feature Importance Analysis</h3></div>', unsafe_allow_html=True)
    
    if 'Attrition' in df.columns:
        # Prepare data for feature importance
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Train Random Forest for feature importance
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_encoded, y)

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        # Select top features for visualization
        top_features = importance_df.tail(15)  # Top 15 features
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top 15 Most Important Features")
            
            fig = go.Figure(go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                marker_color='#2c5aa0',
                text=np.round(top_features['Importance'], 3),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Feature Importance Ranking",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Feature Statistics")
            
            total_importance = importance_df['Importance'].sum()
            top_5_importance = importance_df.tail(5)['Importance'].sum()
            
            st.markdown(f"""
            <div class="success-box">
                <h4>Importance Summary</h4>
                <p><strong>Total Features:</strong> {len(importance_df)}</p>
                <p><strong>Top 5 Features:</strong> {top_5_importance*100:.1f}% of total importance</p>
                <p><strong>Most Important:</strong> {importance_df.iloc[-1]['Feature']}</p>
                <p><strong>Score:</strong> {importance_df.iloc[-1]['Importance']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature categories
            st.subheader("Feature Categories")
            categorical_features = len([col for col in X.columns if X[col].dtype == 'object'])
            numerical_features = len([col for col in X.columns if X[col].dtype in ['int64', 'float64']])
            
            st.markdown(f"""
            <div class="info-box">
                <h4>Feature Types</h4>
                <p><strong>Categorical:</strong> {categorical_features}</p>
                <p><strong>Numerical:</strong> {numerical_features}</p>
                <p><strong>After Encoding:</strong> {len(X_encoded.columns)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download feature importance data
            csv_data = importance_df.to_csv(index=False)
            st.download_button(
                label="Download Feature Importance Data",
                data=csv_data,
                file_name="feature_importance.csv",
                mime="text/csv"
            )
    else:
        st.markdown("""
        <div class="warning-box">
            <h4>Target Variable Not Found</h4>
            <p>Feature importance analysis requires a target variable named 'Attrition'.</p>
        </div>
        """, unsafe_allow_html=True)

def show_model_performance_page(df):
    st.header("Model Performance Analysis")
    
    # Remove columns that weren't used in training
    columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Prepare data
    X = df_clean.drop(columns=['Attrition'])  # Features
    y = df_clean['Attrition']  # Target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load models
    try:
        trained_models = load_models()
        if not trained_models:
            st.warning("No pre-trained models found. Please train models first or provide pre-trained model files in a 'Models' directory.")
            st.info("The 'Models' directory should contain .joblib files with trained models.")
            return
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Calculate metrics
    evaluation_results = []
    
    for model_name, model in trained_models.items():
        try:
            # Skip helper models
            if model_name in ["CatBoost", "Random_Forest"]:
                continue
            
            # Make predictions based on model type
            if isinstance(model, CascadeWrapper):
                y_pred = model.predict(X_test)
            elif isinstance(model, CatBoostKNNWrapper):
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            evaluation_results.append({
                "Model Name": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })
            
        except Exception as e:
            st.warning(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    # Check if we have any results
    if not evaluation_results:
        st.error("No models could be successfully evaluated.")
        return
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(evaluation_results)
    
    # Display metrics table
    st.subheader("Model Performance Metrics")
    st.dataframe(results_df.round(3))
    
    # Create visualization
    st.subheader("Performance Metrics Visualization")
    
    # Create tabs for different visualization options
    tab1, tab2 = st.tabs(["Individual Metrics", "Comparative View"])
    
    with tab1:
        # Individual metric plots
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        
        for metric in metrics:
            fig = go.Figure(data=[
                go.Bar(
                    x=results_df["Model Name"],
                    y=results_df[metric],
                    text=results_df[metric].round(3),
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"{metric} by Model",
                xaxis_title="Model",
                yaxis_title=metric,
                yaxis_range=[0, 1],
                height=400
            )
            
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Only include models with complete data for parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=results_df.index),
                dimensions=[
                    dict(range=[0, 1], label="Accuracy", values=results_df["Accuracy"]),
                    dict(range=[0, 1], label="Precision", values=results_df["Precision"]),
                    dict(range=[0, 1], label="Recall", values=results_df["Recall"]),
                    dict(range=[0, 1], label="F1 Score", values=results_df["F1 Score"])
                ]
            )
        )
        
        fig.update_layout(
            title="Comparative Model Performance",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add download button for the results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Performance Metrics CSV",
        data=csv,
        file_name="model_performance_metrics.csv",
        mime="text/csv"
    )

    # Display model information
    st.subheader("Model Information")
    for model_name in results_df["Model Name"]:
        with st.expander(f"About {model_name}"):
            if model_name == "Stacked RF+GB+SVM":
                st.write("A stacked ensemble combining Random Forest, Gradient Boosting, and Support Vector Machine. Stacking ensembles, or stacked generalization, is a robust ensemble learning technique that combines the strengths of multiple predictive models to enhance overall performance. Unlike methods such as bagging and boosting, which often rely on homogeneous models, stacking leverages a diverse set of algorithms, each specializing in different aspects of the data. ")
            elif model_name == "Cascading Classifiers":
                st.write("A Cascading Classifier combines classifiers in a sequential pipeline to refine predictions progressively, leveraging the strengths of different algorithms. In this study, a Random Forest classifier is used as the initial base classifier, followed by Gradient Boosting as the second classifier. This combination is particularly effective in employee attrition prediction due to the complementary strengths of these two algorithms.")
            elif model_name == "Calibration Curves":
                st.write("Random Forest classifier and Logistic Regression are combined using calibration methods like platt scaling and isotonic regression to enhance the probability estimation accuracy of the model. The Random Forest classifier serves as the primary base model. Logistic Regression applied in Platt Scaling, refines these probabilities by fitting a logistic regression model to the Random Forest’s outputs. This creates a smoother and more reliable probability distribution. The use of calibration curves in this context evaluates the alignment between the model’s predicted probabilities and real-world attrition outcomes.")
            elif model_name == "HGBoost+KNN":
                st.write("A hybrid model combining Histogram-based Gradient Boosting with k-Nearest Neighbors. The Histogram-based Gradient Boosting Classification Tree (Hist Gradient Boosting Classifier) is an advanced machine learning algorithm that enhances predictive performance by discretizing continuous features into discrete bins, thus reducing computational complexity and memory usage.")
            elif model_name == "XGBRF":
                st.write("XGBoost with Random Forest-like tree growing. XGBoost (eXtreme Gradient Boosting) is a powerful gradient boosting framework known for its speed and performance. It has been applied in employee attrition analysis to determine the root causes of employee resignation, with studies indicating that XG-Boost can effectively model complex relationships within HR data.")
            elif model_name == "CatBoost+KNN":
                st.write("A hybrid model combining CatBoost with k-Nearest Neighbors. CatBoost is a gradient boosting algorithm specifically designed to handle categorical features efficiently without extensive pre-processing.")
            elif model_name == "Bagging":
                st.write("Bagging (Bootstrap Aggregating) is an ensemble learning technique that reduces variance and avoids overfitting by training multiple models on different random subsets of the training data and combining their predictions. This model uses a DecisionTreeClassifier as the base learner. Each tree is trained on a different bootstrap sample of the training data.")

# 2. Show the user interface and make predictions
def show_prediction_interface(trained_models):
    st.markdown("""
    <div class="main-header">
        <h1>Employee Attrition Prediction</h1>
        <p>Advanced machine learning models for workforce retention prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are available
    if not trained_models:
        st.markdown("""
        <div class="warning-box">
            <h4>Prediction Models Not Available</h4>
            <p>No pre-trained models are currently loaded. To enable predictions, please ensure the following:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h3>Required Model Files</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>Primary Models</h4>
                <ul>
                    <li>Stacked_RF+GB+SVM.joblib</li>
                    <li>Cascading_Classifiers.joblib</li>
                    <li>Calibration_Curves.joblib</li>
                    <li>HGBoost+KNN.joblib</li>
                    <li>XGBRF.joblib</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>Supporting Models</h4>
                <ul>
                    <li>CatBoost+KNN.joblib</li>
                    <li>CatBoost.joblib</li>
                    <li>Random_Forest.joblib</li>
                    <li>Bagging.joblib</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Setup Instructions</h4>
            <p>1. Create a 'Models' directory in your project folder</p>
            <p>2. Place the trained model files (.joblib format) in this directory</p>
            <p>3. Restart the application to load the models</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Split the screen into two columns for better readability
    col1, col2 = st.columns(2)

    # ------------------------
    # Column 1: Numeric inputs
    # ------------------------
    with col1:
        age = st.slider("Age", 18, 65, 30)
        daily_rate = st.number_input("Daily Rate", min_value=1, max_value=3000, value=800)
        distance = st.slider("Distance from Home", 0, 30, 10)
        education = st.slider("Education (1=Low, 4=High)", 1, 4, 2)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
        job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
        performance_rating = st.slider("Performance Rating (1=Low, 4=High)", 1, 4, 3)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        hourly_rate = st.number_input("Hourly Rate", min_value=10, max_value=100, value=20)

    # ------------------------
    # Column 2: Categorical inputs
    # ------------------------
    with col2:
        business_travel = st.selectbox("Business Travel",
                                       ["Non-Travel", "Travel Frequently", "Travel Rarely"])
        department = st.selectbox("Department",
                                  ["Human Resources", "Research & Development", "Sales"])
        marital_status = st.selectbox("Marital Status",
                                      ["Divorced", "Married", "Single"])
        education_field = st.selectbox("Education Field",
                                       ["Human Resources", "Life Sciences", "Marketing",
                                        "Medical", "Other", "Technical Degree"])
        job_role = st.selectbox("Job Role",
                                ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                                 "Manager", "Manufacturing Director", "Research Director",
                                 "Research Scientist", "Sales Executive", "Sales Representative"])
        gender = st.selectbox("Gender (Male=1, Female=0)", ["Male", "Female"])
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 40, 5)

    # OverTime selection (placed outside the two columns)
    overtime = st.selectbox("OverTime", ["Yes", "No"])

    # 3. Build the final input dictionary with EXACT columns your model expects
    input_data = {
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance],
        'Education': [education],
        'EmployeeNumber': [12345],  # Arbitrary or user-provided
        'EnvironmentSatisfaction': [3],  # Default or you can add another slider
        'Gender': [1 if gender == "Male" else 0],
        'HourlyRate': [hourly_rate],
        'JobInvolvement': [3],          # Default or add another slider
        'JobLevel': [3],                # Default or add another slider
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'MonthlyRate': [10000],         # Default or let user input
        'NumCompaniesWorked': [5],      # Default or let user input
        'OverTime': [1 if overtime == "Yes" else 0],
        'PercentSalaryHike': [10],      # Default or let user input
        'PerformanceRating': [performance_rating],
        'RelationshipSatisfaction': [3],
        'StockOptionLevel': [stock_option_level],
        'TotalWorkingYears': [total_working_years],
        'TrainingTimesLastYear': [2],
        'WorkLifeBalance': [3],
        'YearsAtCompany': [years_at_company],
        'YearsInCurrentRole': [years_in_current_role],
        'YearsSinceLastPromotion': [1],  # Default
        'YearsWithCurrManager': [2],     # Default

        # One-hot columns for BusinessTravel
        'BusinessTravel_Non-Travel': [1 if business_travel == "Non-Travel" else 0],
        'BusinessTravel_Travel_Frequently': [1 if business_travel == "Travel Frequently" else 0],
        'BusinessTravel_Travel_Rarely': [1 if business_travel == "Travel Rarely" else 0],

        # One-hot columns for Department
        'Department_Human Resources': [1 if department == "Human Resources" else 0],
        'Department_Research & Development': [1 if department == "Research & Development" else 0],
        'Department_Sales': [1 if department == "Sales" else 0],

        # One-hot columns for Marital Status
        'MaritalStatus_Divorced': [1 if marital_status == "Divorced" else 0],
        'MaritalStatus_Married': [1 if marital_status == "Married" else 0],
        'MaritalStatus_Single': [1 if marital_status == "Single" else 0],

        # One-hot columns for EducationField
        'EducationField_Human Resources': [1 if education_field == "Human Resources" else 0],
        'EducationField_Life Sciences': [1 if education_field == "Life Sciences" else 0],
        'EducationField_Marketing': [1 if education_field == "Marketing" else 0],
        'EducationField_Medical': [1 if education_field == "Medical" else 0],
        'EducationField_Other': [1 if education_field == "Other" else 0],
        'EducationField_Technical Degree': [1 if education_field == "Technical Degree" else 0],

        # One-hot columns for JobRole
        'JobRole_Healthcare Representative': [1 if job_role == "Healthcare Representative" else 0],
        'JobRole_Human Resources': [1 if job_role == "Human Resources" else 0],
        'JobRole_Laboratory Technician': [1 if job_role == "Laboratory Technician" else 0],
        'JobRole_Manager': [1 if job_role == "Manager" else 0],
        'JobRole_Manufacturing Director': [1 if job_role == "Manufacturing Director" else 0],
        'JobRole_Research Director': [1 if job_role == "Research Director" else 0],
        'JobRole_Research Scientist': [1 if job_role == "Research Scientist" else 0],
        'JobRole_Sales Executive': [1 if job_role == "Sales Executive" else 0],
        'JobRole_Sales Representative': [1 if job_role == "Sales Representative" else 0]
    }

    # 4. Convert to DataFrame (same columns used in training!)
    input_df = pd.DataFrame(input_data)

    # 5. Predict using all models
    if st.button("Predict"):
        predictions = {}
        for model_name, model in trained_models.items():
            try:
                if model_name == "CatBoost":
                    # CatBoost-specific prediction logic
                    input_df['CategoricalColumn'] = input_df['CategoricalColumn'].astype(str)  # Replace with actual categorical columns
                    pred = model.predict(input_df)
                else:
                    pred = model.predict(input_df)
                predictions[model_name] = "Yes" if pred[0] == 1 else "No"
            except Exception as e:
                predictions[model_name] = f"Error: {str(e)}"

        # Display predictions
        st.subheader("Predictions from all models:")
        for model_name, prediction in predictions.items():
            st.write(f"{model_name}: {prediction}")

# Main application
def main():
    st.markdown("""
    <div class="main-header">
        <h1>Workforce Analytics Dashboard</h1>
        <p>Comprehensive Employee Data Analysis & Attrition Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    df_processed = preprocess_data(df)

    # Enhanced sidebar with styling
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">Navigation Menu</h3>
        <p style="font-size: 0.9em; color: #718096;">Select an analysis module below</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose Analysis Module",
        ["Overview", "Data Exploration", "PCA Analysis","Feature Analysis", "Model Performance", "Prediction Interface"],
        help="Navigate through different analysis modules"
    )
    
    # Add information in sidebar
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h4 style="color: #667eea; font-weight: 600;">Module Information</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Overview":
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>Overview:</strong> View key workforce metrics and distribution analysis</p>
        </div>
        """, unsafe_allow_html=True)
    elif page == "Data Exploration":
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>Data Exploration:</strong> Explore data patterns and feature distributions</p>
        </div>
        """, unsafe_allow_html=True)
    elif page == "PCA Analysis":
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>PCA Analysis:</strong> Dimensionality reduction and component analysis</p>
        </div>
        """, unsafe_allow_html=True)
    elif page == "Feature Analysis":
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>Feature Analysis:</strong> Feature importance and correlation analysis</p>
        </div>
        """, unsafe_allow_html=True)
    elif page == "Model Performance":
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>Model Performance:</strong> Evaluate machine learning model performance</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="info-box">
            <p style="margin: 0;"><strong>Prediction Interface:</strong> Generate attrition predictions for individual employees</p>
        </div>
        """, unsafe_allow_html=True)

    if page == "Overview":
        show_overview_page(df)
    elif page == "Data Exploration":
        display_data_exploration(df)
    elif page == "PCA Analysis":
        display_pca_analysis(df)
    elif page == "Feature Analysis":
        show_feature_analysis_page(df_processed)
    elif page == "Model Performance":
        show_model_performance_page(df_processed)
    else:
        try:
            trained_models = load_models()
            show_prediction_interface(trained_models)
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.info("Please check that the 'Models' directory exists and contains the required model files.")

if __name__ == "__main__":

    main()


