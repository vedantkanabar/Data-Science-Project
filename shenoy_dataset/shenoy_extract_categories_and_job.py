import pandas as pd

dFrame = pd.read_csv("shenoy_dataset_cleaned.csv")

categories = sorted(dFrame['category'].unique())
jobs = sorted(dFrame['job'].unique())

with open("categories.txt", "w") as file:
    for category in categories:
        file.write(category + '\n')

with open("jobs.txt", "w") as file:
    for job in jobs:
        file.write(job + '\n')