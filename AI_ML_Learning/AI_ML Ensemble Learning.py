# Databricks notebook source
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import re
import logging
import ast
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

def evaluate_dataset_via_ai(dataset):
    """
    Function to evaluate dataset using OpenAI API
    """
    # Convert the dataset to a string format
    dataset_str = dataset.to_csv(index=False)

    example_return_format = "{'selected_features': [list_of_selected_features], 'target_column': target_column}"

    # Create a prompt for the AI to evaluate the dataset
    prompt = f"""[NO PROSE]
    The following is a dataset in CSV format. Please evaluate the dataset and return a list of selected features and a target column for a Random Forest Regression model.

    I want you to plan to use all columns except for the target column as selected features. For example, if you have columns A, B, C, D, E, and you select column C as the target column for this regression, I want you to return a list of columns A, B, D, E as the selected features.

    Please return the data in the following format with no ```json```: {example_return_format}

    Dataset:
    {dataset_str}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    message = response.choices[0].message.content

    return message

def parse_evaluation(evaluation):
    """
    Function to parse AI evaluation and extract preprocessing steps
    """
    eval_dict = ast.literal_eval(evaluation)
    selected_features = eval_dict['selected_features']
    target_column = eval_dict['target_column']

    return selected_features, target_column

def preprocess_dataset(dataset, selected_features, target_column):
    """
    Function to preprocess dataset based on selected features and target column
    """
    dataset = dataset.dropna()  # Drop rows with null values
    
    # Separate features and target
    X = dataset[selected_features].copy()
    y = dataset[target_column]
    
    # Label encode categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    return X, y, label_encoders

# Load your dataset
dataset = pd.read_csv('/Workspace/Users/neil.lewis@credera.com/AI_ML_Learning/Test Data AI_ML - Sheet1.csv')

# Evaluate the dataset using OpenAI API
evaluation = evaluate_dataset_via_ai(dataset)
print(f"AI Evaluation of the dataset:\n{evaluation}\n")

# Parse the AI evaluation to get preprocessing details
selected_features, target_column = parse_evaluation(evaluation)

print(f"Selected features: {selected_features}\n")
print(f"Target column: {target_column}\n")

if not selected_features or not target_column:
    raise ValueError("Failed to extract necessary preprocessing details from AI evaluation.")

# Preprocess the dataset based on AI evaluation
X, y, label_encoders = preprocess_dataset(dataset, selected_features, target_column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate and display feature importance
feature_importances = model.feature_importances_
features = X_train.columns

# Combine feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

# Reverse engineer feature importances for categorical features
for column, le in label_encoders.items():
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label encoding for {column}: {mapping}")

# COMMAND ----------

X

# COMMAND ----------

X_train, X_test, y_train, y_test, y_pred

# COMMAND ----------

# MAGIC %md
# MAGIC Class Based

# COMMAND ----------

import os

os.getcwd()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import logging
from dotenv import load_dotenv
import os
from classes.process_data import PrepData
from classes.random_forest_regression import RandomForestRegression

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_key = os.getenv("OPENAI_API_KEY")
dataset = pd.read_csv('/Workspace/Users/neil.lewis@credera.com/AI_ML_Learning/data/Test Data AI_ML - Sheet1.csv')


# Preprocess the dataset based on AI evaluation
data_prepper = PrepData(dataset, openai_key)
X, y, label_encoders = data_prepper.preprocess_dataset()

# Perform Random Forest Regression
rf_regressor = RandomForestRegression(X, y, label_encoders)
rf_regressor.rf_regression()

# COMMAND ----------


