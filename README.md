# Data-Science-Project

## Flow of Programs

### Cleaning Up the 2 Datasets:
-  `choudry_dataset_cleanup.py`
-  `shenoy_dataset_cleanup.py`

Both of the files above will cleanup the 2 datasets that we have. It will perform cleanup such as making sure the columns on both datasets have the same structure and removing any unnecessary substrings.

### Cleaning and Dombining Datasets
- `combine_dataset.py`

When this script is run, it will combine both datasets into one. This is also where unneccessary columns are being dropped from the dataset.

### Dataset Encoding
- `encode_dataset.py`

This script will perform encodings on non-numerical value from our columns.

- `encoder_map.py`

This script will generate a txt file of the encoding mappings

### Resampling
- `resampling.py`

This script will perform evaluation on resampling methods such as Random Sample, Random Over Sample, SMOTE, ADASYN, Random Under Sample, Near-Miss 1, Near-Miss 2, and Near-Miss 3. This script will also generate metrics for each method.

### Model Training and Evaluation
- `model_training_evaluation.py`

This script will perform model evaluation such as Logistic Regression, Decision Tree, Random Forest, SVM Linear, KNN, Naive Bayes, and Adaboost. This script also handles missing values and fills them with the median using SimpleImputer. This script will also generate metrics for each model.

- `model_tuning.py`

This script is used for hyperparameter tuning using RandomizedSearch.

### Historical Features
- `create_features.py`

This script will create features for transaction history based on credit card number (Only using Shenoy dataset). We generated 1-7 transaction histories.

### Model With History
- `model_evaluation_historical.py`

This script will perform model evaluation using the data with transaction history. For this part, we use the data of 1, 3, 5, and 7 transaction histories.