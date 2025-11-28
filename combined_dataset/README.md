# Combined Dataset

- This folder contains the code for combining the features
- The `combine_dataset.py` file first combines `../shenoy_dataset/shenoy_dataset_cleaned.csv` and `../choudhury_dataset/choudhury_dataset_cleaned.csv` to create `combined.csv`. 
- The `encode_dataset.py` file encodes the label colums and stores them in the `combined.csv` and also stores the Label encoder in `encoderList.pkl`
- The `encoder_map.py` file reads the encoder from `encoderList.pkl` and prints the encoding map for each column into `encoder_mappings.txt`
- **NOTE:** The csv files are also too large so they are zipped with 7z format (https://www.7-zip.org/7z.html).