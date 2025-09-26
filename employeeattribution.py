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

# Page configg
st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path relative to the script
    file_path = os.path.join(base_path, 'employeedata.csv')
    print(f"Loading data from: {file_path}")  # Log file path for debugging
    # Load the dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Numerical encoding
    label_cols = ['Attrition', 'Gender', 'Over18', 'OverTime']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # One-hot encoding
    categorical_cols = ['BusinessTravel', 'Department', 'MaritalStatus', 'EducationField', 'JobRole']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df

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
                raise FileNotFoundError(f"Model file not found: {file_name}")
            
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
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        fig = px.pie(df, names='Attrition', title='Employee Attrition Distribution')
        st.plotly_chart(fig)

    with col2:
        st.subheader("Department-wise Attrition")
        dept_attrition = df.groupby(['Department', 'Attrition']).size().unstack()
        fig = px.bar(dept_attrition, barmode='group', title='Attrition by Department')
        st.plotly_chart(fig)

    st.subheader("Key Metrics")
    metrics = st.columns(4)
    metrics[0].metric("Total Employees", len(df))
    metrics[1].metric("Attrition Rate", f"{(df['Attrition'] == 'Yes').mean():.1%}")
    metrics[2].metric("Avg Tenure", f"{df['YearsAtCompany'].mean():.1f} years")
    metrics[3].metric("Avg Age", f"{df['Age'].mean():.1f} years")
    

def display_data_exploration(df):
    st.subheader("Data Exploration")

    # Dataset overview
    if st.checkbox("Show Dataset"):
        st.write(df)

    # Column analysis
    st.subheader("Column Analysis")
    selected_column = st.selectbox("Select Column to Analyze", df.columns)

    col1, col2 = st.columns(2)

    with col1:
        # Distribution plot
        fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Box plot
        fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
        st.plotly_chart(fig, use_container_width=True)

def display_pca_analysis(df):
    st.subheader("PCA Analysis")

    # Prepare data for PCA
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    n_components = st.slider("Select number of PCA components", 2, min(10, len(numeric_cols)), 3)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # Plot explained variance ratio
    fig = px.line(
        x=range(1, n_components + 1),
        y=pca.explained_variance_ratio_,
        title="Explained Variance Ratio by Principal Components",
        labels={"x": "Principal Component", "y": "Explained Variance Ratio"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2D scatter plot of first two components
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=df['Attrition'],
        title="First Two PCA Components"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_feature_analysis_page(df):
    st.header("Feature Analysis")

    # Correlation heatmap
    st.subheader("Feature Correlation")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, title="Correlation Heatmap")
    st.plotly_chart(fig)

    # Feature importance
    st.subheader("Feature Importance")
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig = px.bar(importance_df.head(10), x='Importance', y='Feature',
                 title='Top 10 Important Features')
    st.plotly_chart(fig)

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
    st.header("Attrition Prediction Interface")

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
    st.title("Employee Attrition Analysis Dashboard")
    df = load_data()
    df_processed = preprocess_data(df)

    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Exploration", "PCA Analysis","Feature Analysis", "Model Performance", "Prediction Interface"]
    )

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
        trained_models = load_models()
        show_prediction_interface(trained_models)

if __name__ == "__main__":

    main()


