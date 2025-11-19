import pandas as pd
import json

# Read the CSV file with optimized parameters for large files
df = pd.read_csv('shenoy_dataset_cleaned.csv', 
                 parse_dates=['trans_date_trans_time'],
                 low_memory=False)
# Sort by credit card number and timestamp to enable historical feature calculation
df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

# Group by credit card number to calculate historical features
grp = df.groupby('cc_num')

# Time since last transaction (in minutes)
df['prev_trans_time'] = (df['trans_date_trans_time'] - grp['trans_date_trans_time'].shift(1)).dt.total_seconds() / 60

# Previous transaction amount
df['amt_prev'] = grp['amt'].shift(1)

# Average amount for this credit card (over all transactions)
df['avg_amt'] = grp['amt'].transform('mean')

# Rolling count of transactions in the last 1 hour
df['trans_count_1h'] = (grp.rolling('1H', on='trans_date_trans_time')['amt'].count().reset_index(level=0, drop=True).values)


# Initialize a list to store JSON data
json_data = []

# Iterate over each row to build the JSON structure
for _, row in df.iterrows():
    
    # Create the base JSON structure with labels, is_fraud and features
    data = {
        "labels": {
            "cc_num": row["cc_num"],
            "timestamp": row["trans_date_trans_time"].isoformat()
        },
        "is_fraud": row["is_fraud"],
        "features": {}
    }
    
    # Add all original columns as features (except labels and target)
    for column in df.columns:
        if column not in ["cc_num", "trans_date_trans_time", "is_fraud"]:
            # Handle datetime columns by converting to ISO format string
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                data["features"][column] = row[column].isoformat()
            else:
                # Add the column value to the "features" object
                # This includes both original columns AND the new historical features
                data["features"][column] = row[column]
    
    # Append the data for the current row to the JSON list
    json_data.append(data)

# Write the JSON data to file
with open('shenoy_features.json', 'w') as f:
    json.dump(json_data, f, indent=2, default=str)
