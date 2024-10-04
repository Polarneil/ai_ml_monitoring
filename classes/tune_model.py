import logging
import ast
import pandas as pd

class TuneModel:
    def __init__(self, OpenAIWrapper):
        self.OpenAIWrapper = OpenAIWrapper

    def analyze_performance(self, mse, r2, accuracy, test_size, model_params, feature_importance_df):
        feature_importance_csv = feature_importance_df.to_csv(index=False)
        
        example_return_format = "{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}"

        prompt = f"""[NO PROSE]
        You will be given a list of performance metric results from a Random Forest Regression Machine Learning Model.

        Your job is to analyze the performance of this model and fine tune the model to improve the accuracy and R^2 score of the model.

        You will do this by tuning any one of the 17 parameters in the model that can be found here:
        {model_params}

        The above parameters resulted in the following performance metrics:
        - Accuracy: {accuracy}
        - R-Squared (R^2): {r2}
        - Test Size: {test_size}
        - Mean Squared Error (MSE): {mse}
        - Feature Importance: {feature_importance_csv}

        Given these results, your only goal is to improve accuracy and R^2 of the model with new parameters and test size.
        
        Your task is to tune the model to improve the accuracy and R^2. You will return a list of JSON objects with no ```json```. The first element in the list will be in the new parameters in the same format that you were given them in above. The second element will be the new test size with the key 'test_size'. It is crucial that you will not include any escape characters in your response.
        """

        message = self.OpenAIWrapper.chat_completion(prompt)

        return message
