import openai
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import logging
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrepData:
    def __init__(self, dataset, openai_key):
        self.dataset = dataset
        self.openai_key = openai_key
        self.client = OpenAI(api_key=self.openai_key)

    def evaluate_dataset_via_ai(self):
        """
        Function to evaluate dataset using OpenAI API
        """
        # Convert the dataset to a string format
        dataset_str = self.dataset.to_csv(index=False)

        example_return_format = "{'selected_features': [list_of_selected_features], 'target_column': target_column}"

        # Create a prompt for the AI to evaluate the dataset
        prompt = f"""[NO PROSE]
        The following is a dataset in CSV format. Please evaluate the dataset and return a list of selected features and a target column for a Random Forest Regression model.

        I want you to plan to use all columns except for the target column as selected features. For example, if you have columns A, B, C, D, E, and you select column C as the target column for this regression, I want you to return a list of columns A, B, D, E as the selected features.

        Please return the data in the following format with no ```json```: {example_return_format}

        Dataset:
        {dataset_str}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        message = response.choices[0].message.content

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
    