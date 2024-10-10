import ast
from classes.rf_regression_optimized import RandomForestRegression


class IterateModel:
    def __init__(self,
                 label_encoders,
                 desired_accuracy,
                 run_threshold,
                 tuner,  # Class
                 performance_tracking_file
                 ):
        self.label_encoders = label_encoders
        self.desired_accuracy = desired_accuracy
        self.run_threshold = run_threshold
        self.tuner = tuner
        self.performance_tracking_file = performance_tracking_file

    def iterate_model(self, X, y, mse, r2, accuracy, test_size, model_params, feature_importance):

        num_runs = 1
        while accuracy < self.desired_accuracy and num_runs != self.run_threshold:
            num_runs += 1

            revised_metrics = ast.literal_eval(self.tuner.analyze_performance(mse, r2, accuracy, test_size, model_params, feature_importance, self.performance_tracking_file))

            rf_regressor = RandomForestRegression(
                X=X,
                y=y,
                num_runs=num_runs,

                bootstrap=revised_metrics[0]['bootstrap'],
                ccp_alpha=revised_metrics[0]['ccp_alpha'],
                criterion=revised_metrics[0]['criterion'],
                max_depth=revised_metrics[0]['max_depth'],
                max_features=revised_metrics[0]['max_features'],
                max_leaf_nodes=revised_metrics[0]['max_leaf_nodes'],
                max_samples=revised_metrics[0]['max_samples'],
                min_impurity_decrease=revised_metrics[0]['min_impurity_decrease'],
                min_samples_leaf=revised_metrics[0]['min_samples_leaf'],
                min_samples_split=revised_metrics[0]['min_samples_split'],
                min_weight_fraction_leaf=revised_metrics[0]['min_weight_fraction_leaf'],
                n_estimators=revised_metrics[0]['n_estimators'],
                n_jobs=revised_metrics[0]['n_jobs'],
                oob_score=revised_metrics[0]['oob_score'],
                random_state=revised_metrics[0]['random_state'],
                verbose=revised_metrics[0]['verbose'],
                warm_start=revised_metrics[0]['warm_start'],

                test_size=revised_metrics[1]['test_size'],
                label_encoders=self.label_encoders
                )

            mse, r2, accuracy, test_size, model_params, feature_importance = rf_regressor.rf_regression()