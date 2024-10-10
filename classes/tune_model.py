import logging
import ast
import pandas as pd
import json

class TuneModel:
    def __init__(self, OpenAIWrapper, prompt_docs):
        self.OpenAIWrapper = OpenAIWrapper
        self.prompt_docs = open(prompt_docs, 'r').read()

    def analyze_performance(self, mse, r2, accuracy, test_size, model_params, feature_importance_df, performance_tracking_file):
        feature_importance_csv = feature_importance_df.to_csv(index=False)
        with open(performance_tracking_file, 'r') as perf_file:
            performance_tracking_text = json.load(perf_file)
        
        example_return_format = "[{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 1.0, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}, {'test_size': 0.30}]"

        prompt = f"""[NO PROSE]
        You will be given a list of performance metric results from a scikit learn Random Forest Regression Machine Learning Model.

        Your job is to analyze the performance of this model and fine tune the model to improve the accuracy of the model to at least 80. Remember that you have the freedom to take risks. This will be evaluated 10 different times so do whatever you need to do to improve the accuracy.

        You will do this by tuning any one of the 17 parameters in the model that can be found here:
        {model_params}


        You will tune these parameters by using the scikit-learn documentation for the parameters found here:
        {self.prompt_docs}


        The above parameters resulted in the following performance metrics:
        - Accuracy: {accuracy}
        - R-Squared (R^2): {r2}
        - Test Size: {test_size}
        - Mean Squared Error (MSE): {mse}
        - Feature Importance: {feature_importance_csv}


        Given these results, your only goal is to improve accuracy of the model with new parameters and test size.
        
        Your task is to tune the model to improve the accuracy. You will return a list of JSON objects with no ```json``` in the sring. This needs to avoid syntax errors. The first element in the list will be the new parameters. The second element will be the new test size with the key 'test_size'. It is crucial that you will not include any escape characters in your response like `\`. Never ever include `\` characters in your response. You will only ever return the json response and nothing more. Never any verbiage [NO PROSE]. Use the following example return format as a reference:
        {example_return_format}


        Do not let the values influence you as they are just defaults. Make sure you always return all 17 parameters provided in the example response format.


        After each run, the results will be wrote to a json file to track your progress. Below you will find this file which contains the performance results from each run along with the parameters and test size. Leverage this and learn from it as you continue to fine tune the parameters. If the accuracy decreased between certain runs, make sure to catch on to that pattern. Likewise, if accuracy has increased between some runs, make sure to catch on to that pattern as well. Your goal will always be to improve accuracy. Performance result json file:
        {performance_tracking_text}
        """

        message = self.OpenAIWrapper.chat_completion(prompt)

        return message
