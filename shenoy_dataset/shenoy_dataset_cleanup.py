import pandas as pd

# Opening the Shenoy Dataset
dFrame = pd.read_csv("shenoy_dataset.csv")

# Removing the text "fraud_" from each of the merchant in the merchant column
dFrame['merchant'] = dFrame['merchant'].str.replace(r'^fraud_', '', regex=True)

# Making every string lowercase to make the dataset uniform
for col in dFrame.columns:
    if dFrame[col].dtype == "object":
        dFrame[col] = dFrame[col].str.lower()

# Write the result to a CSV file
dFrame.to_csv("shenoy_dataset_cleaned.csv", index=False)