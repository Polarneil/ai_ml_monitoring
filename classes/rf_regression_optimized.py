import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomForestRegression:
    def __init__(
        self, 
        X, 
        y,

        n_estimators = 100,
        criterion = 'squared_error',
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.0,
        max_features = 1.0,
        max_leaf_nodes =  None,
        min_impurity_decrease = 0.0,
        bootstrap = True,
        oob_score = False,
        n_jobs = None,
        random_state = 42,
        verbose = 0,
        warm_start = False,
        ccp_alpha = 0.0,
        max_samples = None,

        test_size = 0.30, 
        label_encoders = None,
        feature_importance_df = None,
        mse = None,
        r2 = None,
        accuracy = None,
        ):

        self.X = X
        self.y = y

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        self.test_size = test_size
        self.label_encoders = label_encoders
        self.feature_importance_df = feature_importance_df
        self.mse = mse
        self.r2 = r2
        self.accuracy = accuracy

        self.model = RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes, min_impurity_decrease=self.min_impurity_decrease, bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs, random_state=self.random_state, verbose=self.verbose, warm_start=self.warm_start, ccp_alpha=self.ccp_alpha, max_samples=self.max_samples)

    def rf_regression(self):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)


        # Create and train the Random Forest Regression model
        self.model.fit(X_train, y_train)


        # Make predictions and evaluate the model
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

        # Accuracy
        errors = abs(y_pred - y_test)
        mape = 100 * (errors / y_test)
        self.accuracy = 100 - np.mean(mape)

        # Calculate and display feature importance
        feature_importances = self.model.feature_importances_
        features = X_train.columns


        # Combine feature names and their importance scores
        self.feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})


        # Sort features by importance
        self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)


        # Reverse engineer feature importances for categorical features
        for column, le in self.label_encoders.items():
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"\nLabel encoding for {column}: {mapping}\n")
        
        logger.info(f"\n{self.return_model_params()}\n")

        return self.return_model_params()
    
    def return_model_params(self):
        # Return the results paired with the model parameters to further pass into an AI prompt for model tuning

        return self.mse, self.r2, self.accuracy, self.test_size, self.model.get_params(), self.feature_importance_df
