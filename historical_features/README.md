# Historical Features

- The file `create_features.py` creates csv files containing features with history from the `../shenoy_dataset/shenoy_dataset_cleaned.csv` dataset.
- File `i_historical_features.csv` contains features with a history of `i` transactions and `i_historical_features_encoders.pkl` contains the label encoding used.
- **NOTE:** the file creates features with a history of upto 10 transactions but we are only using History for 1,3,5,7 transactions. The csv files are also too large so they are zipped with 7z format (https://www.7-zip.org/7z.html).