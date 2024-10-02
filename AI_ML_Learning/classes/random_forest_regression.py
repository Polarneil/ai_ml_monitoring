import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestRegression:
    def __init__(self, X, y, label_encoders):
        self.X = X
        self.y = y
        self.label_encoders = label_encoders
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def rf_regression(self):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


        # Create and train the Random Forest Regression model
        self.model.fit(X_train, y_train)


        # Make predictions and evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"\nMean Squared Error: {mse}\n")


        # Calculate and display feature importance
        feature_importances = self.model.feature_importances_
        features = X_train.columns


        # Combine feature names and their importance scores
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})


        # Sort features by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        logger.info(f"\nFeature Importances:\n{feature_importance_df}\n")


        # Reverse engineer feature importances for categorical features
        for column, le in self.label_encoders.items():
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"\nLabel encoding for {column}: {mapping}\n")
