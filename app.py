import streamlit as st
from data_handler import load_dataset
from model_handler import train_model
from utils.helpers import clear_session_state
import os
import time

# Mapping problem types to valid evaluation metrics
METRICS_BY_TYPE = {
    "binary classification": [
        "accuracy", "balanced_accuracy", "mcc", "roc_auc", "log_loss",
        "f1", "precision", "recall", "average_precision"
    ],
    "multi-class classification": [
        "accuracy", "balanced_accuracy", "mcc", "log_loss",
        "f1_macro", "precision_macro", "recall_macro"
    ],
    "regression": [
        "r2", "mean_squared_error", "root_mean_squared_error", "mean_absolute_error",
        "median_absolute_error", "mean_absolute_percentage_error"
    ]
}

# Load custom CSS for styling (optional)
def load_css():
    css_path = os.path.join("static", "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AutoML App with AutoGluon",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_css()
    
    st.title("AutoML Application with AutoGluon")

    # Initialize session state variables
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'training_job' not in st.session_state:
        st.session_state.training_job = None
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    
    # 1. Dataset Upload
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)", 
        type=["csv"],
        help="Upload your tabular dataset in CSV format"
    )
    
    if uploaded_file is not None:
        df, error = load_dataset(uploaded_file)
        if error:
            st.error(error)
        else:
            st.session_state.dataset = df
            st.success("Dataset loaded successfully!")
    
    # 2. Show dataset preview and UI for training parameters
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        with st.expander("Dataset Preview (First 10 Rows)"):
            st.dataframe(df.head(10))
        
        # Target column selection
        target_col = st.selectbox(
            "Select Target Column",
            options=df.columns,
            index=len(df.columns) - 1,
            help="Select the column you want to predict"
        )
        
        # Problem type selection
        problem_type = st.selectbox(
            "Select Problem Type",
            options=["binary classification", "multi-class classification", "regression"],
            index=0,
            help="Choose the type of ML problem"
        )
        
        # Metric selection based on problem type
        metric_options = METRICS_BY_TYPE[problem_type]
        eval_metric = st.selectbox(
            "Select Evaluation Metric",
            options=metric_options,
            index=0,
            help="Metric to optimize during training"
        )
        
        # Train-test split slider
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data reserved for testing"
        )
        
        # Time limit slider
        time_limit = st.slider(
            "Training Time Limit (minutes)",
            min_value=1,
            max_value=120,
            value=10,
            help="Maximum time allowed for training"
        )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Training", use_container_width=True):
                if target_col is None:
                    st.warning("Please select a target column!")
                else:
                    with st.spinner("Starting training session..."):
                        try:
                            st.session_state.training_job = train_model(
                                df,
                                target_col,
                                time_limit * 60,  # seconds
                                test_size / 100,
                                eval_metric
                            )
                            st.session_state.training_started = True
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
        with col2:
            if st.button("🧹 Clear Session", use_container_width=True):
                clear_session_state()
                st.experimental_rerun()
        
        # Show training progress and results
        if st.session_state.training_job:
            job = st.session_state.training_job
            
            st.subheader("Training Results")
            
            if not job['complete']:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while not job['complete']:
                    elapsed = time.time() - job['start_time']
                    remaining = max(0, job['time_limit'] - elapsed)
                    progress = min(1.0, elapsed / job['time_limit'])
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                        **Training Status**: {job['status']}  
                        **Elapsed Time**: {int(elapsed)} seconds  
                        **Estimated Remaining**: {int(remaining)} seconds
                    """)
                    time.sleep(1)
                    if progress >= 1.0:
                        break
                
                progress_bar.empty()
                status_text.empty()
            
            if job['complete']:
                if job.get('error'):
                    st.error(f"Training error: {job['error']}")
                else:
                    st.success("Training completed successfully!")
                    with st.expander("Model Leaderboard"):
                        st.dataframe(job['leaderboard'], use_container_width=True)
                    st.plotly_chart(job['performance_plot'], use_container_width=True)
    else:
        st.info("Please upload a dataset to get started.")

if __name__ == "__main__":
    main()
