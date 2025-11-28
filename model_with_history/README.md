# Model with History

- This folder contains code for creating the model with history.
- The `model_evaluation_historical.py` file contains code to read the features from `../historical_features/i_historical_features.csv` for `i =1,3,5,7` and train the model based on the historical features.
- It creates a seperate folder for each history level called `i_historical_result` to store the results.
- The numeric metrics are in `results.txt` and the confusion plots are in files `Confusion Matrix - <Model> - i Historical Features.png` and the feature importance plots are in `Feature_Importance_<Model>_i_Historical.png`.