# Databricks notebook source
from classes.process_data import PrepData
from classes.random_forest_regression import RandomForestRegression

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from dotenv import load_dotenv
import pandas as pd
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_key = os.getenv("OPENAI_API_KEY")
dataset = pd.read_csv('../data/Test Data AI_ML - Sheet1.csv')


# Preprocess the dataset based on AI evaluation
data_prepper = PrepData(dataset, openai_key)
X, y, label_encoders = data_prepper.preprocess_dataset()

# Perform Random Forest Regression
rf_regressor = RandomForestRegression(X, y, label_encoders)
rf_regressor.rf_regression()
