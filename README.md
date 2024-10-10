### Set Up

* After cloning the repo into your local environment you will need to:
    * Create a virtual environment in the project's root directory and run the command `pip install -r requirements.txt`
    * Create a `.env` file in the parent directory and define the variable `OPENAI_API_KEY`
    * Navigate to the notebooks folder and run the `AI_ML_Ensemble_Learning` notebook

### Testing

* Any csv file can be added to the `data` directory
    * The path must be refactored in the `AI_ML_Ensemble_Learning` notebook to point to the new dataset
* After running the notebook, navigate to the `PerformanceTracking.json` file in the `data` directory to see how AI tuned the model during the loop
  * At the beginning of each run, `PerformanceTracking.json` will be cleared

### Code

* Majority of logic is handled in the `classes` folder
    * The notebook is used to call the methods in these classes