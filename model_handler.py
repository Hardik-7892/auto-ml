from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import plotly.express as px
import time
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def train_model(df, target_col, time_limit, test_size, eval_metric):
    """Train AutoGluon model and return results"""
    job = {
        'start_time': time.time(),
        'time_limit': time_limit,
        'complete': False,
        'status': 'Initializing',
        'leaderboard': None
    }
    
    try:
        # Prepare data
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )
        
        # Initialize predictor
        predictor = TabularPredictor(
            label=target_col,
            eval_metric=eval_metric,
            verbosity=1
        )
        
        # Train model
        job['status'] = 'Training models'
        predictor.fit(
            train_data=train_data,
            presets='medium_quality',
            time_limit=time_limit,
            verbosity=0
        )
        
        # Generate results
        job['status'] = 'Generating leaderboard'
        leaderboard = predictor.leaderboard(test_data)
        
        # Create performance visualization
        performance_df = leaderboard[['model', 'score_val']].dropna()
        fig = px.bar(
            performance_df,
            x='model',
            y='score_val',
            title='Model Performance Comparison'
        )
        
        job.update({
            'complete': True,
            'leaderboard': leaderboard,
            'performance_plot': fig,
            'predictor': predictor
        })
        
    except Exception as e:
        job.update({
            'complete': True,
            'error': str(e)
        })
    
    return job
