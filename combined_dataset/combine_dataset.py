import pandas as pd

df_shenoy = pd.read_csv("../shenoy_dataset/shenoy_dataset_cleaned.csv")
df_choudhury = pd.read_csv("../choudhury_dataset/choudhury_dataset_cleaned.csv")

combined = pd.concat([df_choudhury, df_shenoy], ignore_index=True)
combined = combined.drop(columns=["Unnamed: 0", "street", "zip", "cc_num", "first", "last", "trans_num"], errors="ignore")

combined.to_csv("combined.csv", index=False)