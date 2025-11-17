import pandas as pd

df = pd.read_csv("dataset2cleaned.csv", dtype=str)

# This was used to remove double quotes and make everything lower case
# df = df.applymap(
#     lambda x: x.strip('"').lower() if isinstance(x, str) else x
# )

# This was used to modify date format to be the same as shenoy's dataset
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], format="%d-%m-%Y %H:%M", errors="coerce")
df["trans_date_trans_time"] = df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

df["dob"] = pd.to_datetime(df["dob"], format="%d-%m-%Y", errors="coerce")
df["dob"] = df["dob"].dt.strftime("%Y-%m-%d")
df.to_csv("dataset2cleaned2.csv", index=False)

# Printing the unique categories and jobs just to compare with shenoy's dataset
# categories = sorted(df["category"].unique())
# jobs = sorted(df["job"].unique())

# with open("categories.txt", "w") as file:
#     for category in categories:
#         file.write(f"{category}\n")

# with open("jobs.txt", "w") as file:
#     for job in jobs:
#         file.write(f"{job}\n")

