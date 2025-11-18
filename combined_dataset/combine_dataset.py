import pandas as pd

df_shenoy = pd.read_csv("../shenoy_dataset/shenoy_dataset_cleaned.csv")
df_choudhury = pd.read_csv("../choudhury_dataset/dataset2cleaned2.csv")

combined = pd.concat([df_choudhury, df_shenoy], ignore_index=True)
combined = combined.drop(columns=["Unnamed: 0", "street", "zip", "cc_num", "first", "last"], errors="ignore")

combined.to_csv("combined.csv", index=False)