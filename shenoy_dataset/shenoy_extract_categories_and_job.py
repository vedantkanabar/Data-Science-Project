import pandas as pd

# Opening the cleaned dataset
dFrame = pd.read_csv("shenoy_dataset_cleaned.csv")

# Get the unique values for categories and jobs
categories = sorted(dFrame['category'].unique())
jobs = sorted(dFrame['job'].unique())

# Writing those categories to the categories.txt file
with open("categories.txt", "w") as file:
    for category in categories:
        file.write(category + '\n')

# Writing those jobs to the jobs.txt file
with open("jobs.txt", "w") as file:
    for job in jobs:
        file.write(job + '\n')