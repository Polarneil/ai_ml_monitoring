from sklearn.preprocessing import LabelEncoder
import logging
import ast
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepData:
    def __init__(self, dataset, OpenAIWrapper, performance_tracking_file):
        self.dataset = dataset
        self.OpenAIWrapper = OpenAIWrapper
        self.performance_tracking_file = performance_tracking_file

    def reset_performance_tracking(self):
        if os.path.exists(self.performance_tracking_file):
            os.remove(self.performance_tracking_file)
            logger.info("Performance tracking file reset. Proceeding...")
        else:
            logger.info("Performance tracking file does not exist. Proceeding...")

    def evaluate_dataset_via_ai(self):
        """
        Function to evaluate dataset using OpenAI API
        """
        # Take a subset of the data to cut down on token count
        sample_dataset = self.dataset.head(5)

        # Convert the dataset to string format for the prompt
        dataset_str = sample_dataset.to_csv(index=False)

        example_return_format = "{'selected_features': [list_of_selected_features], 'target_column': target_column}"

        # Prompt for the AI to evaluate the dataset
        prompt = f"""[NO PROSE]
        The following is a dataset in CSV format. Please evaluate the dataset and return a list of selected features and a target column for a Random Forest Regression model.

        I want you to plan to use all columns except for the target column as selected features. For example, if you have columns A, B, C, D, E, and you select column C as the target column for this regression, I want you to return a list of columns A, B, D, E as the selected features.

        Please return the data in the following format with no ```json```: {example_return_format}

        Dataset:
        {dataset_str}
        """

        message = self.OpenAIWrapper.chat_completion(prompt)

        return message

    def parse_evaluation(self):
        """
        Function to parse AI evaluation and extract preprocessing steps
        """
        evaluation = self.evaluate_dataset_via_ai()
        logger.info(f"\nAI Evaluation of the dataset:\n{evaluation}\n")

        eval_dict = ast.literal_eval(evaluation)
        selected_features = eval_dict['selected_features']
        target_column = eval_dict['target_column']

        logger.info(f"\nSelected features: {selected_features}\n")
        logger.info(f"\nTarget column: {target_column}\n")

        if not selected_features or not target_column:
            raise ValueError("\nFailed to extract necessary preprocessing details from AI evaluation.\n")

        return selected_features, target_column
    
    def preprocess_dataset(self):
        """
        Function to preprocess dataset based on selected features and target column
        """
        self.reset_performance_tracking()
        selected_features, target_column = self.parse_evaluation()

        dataset = self.dataset.dropna()  # Drop rows with null values
        
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
    