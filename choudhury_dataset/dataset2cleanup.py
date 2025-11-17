import pandas as pd

df = pd.read_csv("dataset2cleaned.csv", dtype=str)

# This was used to remove double quotes and make everything lower case
# df = df.applymap(
#     lambda x: x.strip('"').lower() if isinstance(x, str) else x
# )

# df.to_csv("dataset2cleaned.csv", index=False)

categories = sorted(df["category"].unique())
jobs = sorted(df["job"].unique())

with open("categories.txt", "w") as file:
    for category in categories:
        file.write(f"{category}\n")

with open("jobs.txt", "w") as file:
    for job in jobs:
        file.write(f"{job}\n")

