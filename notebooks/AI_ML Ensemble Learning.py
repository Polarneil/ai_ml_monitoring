# Databricks notebook source
from classes.process_data import PrepData
#from classes.random_forest_regression import RandomForestRegression
from classes.rf_regression_optimized import RandomForestRegression

from dotenv import load_dotenv
import pandas as pd
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_key = os.getenv("OPENAI_API_KEY")
#dataset = pd.read_csv('../data/Test Data AI_ML - Sheet1.csv')
dataset = pd.read_csv('../data/Restaurant_revenue (1).csv')

# Preprocess the dataset based on AI evaluation
data_prepper = PrepData(dataset, openai_key)
X, y, label_encoders = data_prepper.preprocess_dataset()

# Perform Random Forest Regression
rf_regressor = RandomForestRegression(
    X=X, 
    y=y, 
    label_encoders=label_encoders
    )

rf_regressor.rf_regression()
