# Shenoy Dataset

- This is our simulated dataset.
- The file `shenoy_dataset_cleanup.py` creates a cleaned `shenoy_dataset_cleaned.csv` from `shenoy_dataset.csv`. This cleans up some of the formats in the data.
- **Note:** the actual csv are too large and istead are here after being zipped with the 7z format (https://www.7-zip.org/7z.html) and hence `shenoy_dataset.csv` and `shenoy_dataset_cleaned.csv` are in `shenoy_dataset.7z` and `shenoy_dataset_cleaned.7z`respectively.
- The file `shenoy_extract_categories_and_job.py` creates two files `categories.txt` and `jobs.txt` containing the uniqe categories and jobs respectively.
- There is also a file `create_features.py` which was our initial attempt at creating features with history. This file was **not** used in our code. 