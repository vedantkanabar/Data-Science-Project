import pandas as pd

dFrame = pd.read_csv("shenoy_dataset.csv")

dFrame['merchant'] = dFrame['merchant'].str.replace(r'^fraud_', '', regex=True)

for col in dFrame.columns:
    if dFrame[col].dtype == "object":
        dFrame[col] = dFrame[col].str.lower()

dFrame.to_csv("shenoy_dataset_cleaned.csv", index=False)