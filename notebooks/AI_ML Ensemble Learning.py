# Databricks notebook source
from classes.openai_wrapper import OpenAIWrapper
from classes.process_data import PrepData
from classes.rf_regression_optimized import RandomForestRegression
from classes.tune_model import TuneModel

from dotenv import load_dotenv
import pandas as pd
import logging
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
tuner = TuneModel(openai_wrapper)

revised_metrics = tuner.analyze_performance(mse, r2, accuracy, test_size, model_params, feature_importance)
