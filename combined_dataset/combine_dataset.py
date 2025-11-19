import pandas as pd

# Opening the Shenoy and Choudhury dataset
df_shenoy = pd.read_csv("../shenoy_dataset/shenoy_dataset_cleaned.csv")
df_choudhury = pd.read_csv("../choudhury_dataset/choudhury_dataset_cleaned.csv")

# Combining both datasets and dropping unnecessary columns
combined = pd.concat([df_choudhury, df_shenoy], ignore_index=True)
combined = combined.drop(columns=["Unnamed: 0", "street", "zip", "cc_num", "first", "last", "trans_num"], errors="ignore")

# Writing the result to combined.csv file
combined.to_csv("combined.csv", index=False)