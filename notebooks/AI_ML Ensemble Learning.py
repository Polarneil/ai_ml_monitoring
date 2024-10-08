# Databricks notebook source
from classes.openai_wrapper import OpenAIWrapper
from classes.process_data import PrepData
from classes.rf_regression_optimized import RandomForestRegression
from classes.tune_model import TuneModel

from dotenv import load_dotenv
import pandas as pd
import logging
import ast
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initiate OpenAI Wrapper Class
openai_key = os.getenv("OPENAI_API_KEY")

openai_wrapper = OpenAIWrapper(openai_key)


# Preprocess the dataset based on AI evaluation
dataset = pd.read_csv('../data/Restaurant_revenue (1).csv')

data_prepper = PrepData(dataset, openai_wrapper)
X, y, label_encoders = data_prepper.preprocess_dataset()


# Perform Random Forest Regression
rf_regressor = RandomForestRegression(
    X=X, 
    y=y, 
    label_encoders=label_encoders
    )

# First round model results:
mse, r2, accuracy, test_size, model_params, feature_importance = rf_regressor.rf_regression()


# Now we have run the first regression and can enter feedback loop to tune model parameters
prompt_docs = "../docs/RandomForestRegressorDocs.txt"
performance_tracking_file = "../docs/PerformanceTracking.json"

num_runs = 1
while accuracy < 80 and num_runs != 10:
    num_runs += 1
    tuner = TuneModel(openai_wrapper, prompt_docs)

    revised_metrics = ast.literal_eval(tuner.analyze_performance(mse, r2, accuracy, test_size, model_params, feature_importance, performance_tracking_file))

    rf_regressor = RandomForestRegression(
        X=X,
        y=y,
        num_runs=num_runs,

        bootstrap=revised_metrics[0]['bootstrap'],
        ccp_alpha=revised_metrics[0]['ccp_alpha'],
        criterion=revised_metrics[0]['criterion'],
        max_depth=revised_metrics[0]['max_depth'],
        max_features=revised_metrics[0]['max_features'],
        max_leaf_nodes=revised_metrics[0]['max_leaf_nodes'],
        max_samples=revised_metrics[0]['max_samples'],
        min_impurity_decrease=revised_metrics[0]['min_impurity_decrease'],
        min_samples_leaf=revised_metrics[0]['min_samples_leaf'],
        min_samples_split=revised_metrics[0]['min_samples_split'],
        min_weight_fraction_leaf=revised_metrics[0]['min_weight_fraction_leaf'],
        n_estimators=revised_metrics[0]['n_estimators'],
        n_jobs=revised_metrics[0]['n_jobs'],
        oob_score=revised_metrics[0]['oob_score'],
        random_state=revised_metrics[0]['random_state'],
        verbose=revised_metrics[0]['verbose'],
        warm_start=revised_metrics[0]['warm_start'],

        test_size=revised_metrics[1]['test_size'],
        label_encoders=label_encoders
        )

    mse, r2, accuracy, test_size, model_params, feature_importance = rf_regressor.rf_regression()
