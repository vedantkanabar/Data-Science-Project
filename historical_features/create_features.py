import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Read the CSV file with optimized parameters for large files
df = pd.read_csv('../shenoy_dataset/shenoy_dataset_cleaned.csv', 
                 low_memory=False)

# Convert to datetime for calculations (needed for time differences)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# Sort by credit card number and timestamp to enable historical feature calculation
df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

# Group by credit card number
grp = df.groupby('cc_num')

# number of historical transactions to look back
N_HISTORICAL = 10

historical_features = {
    'merchant': 'merchant',
    'merch_lat': 'merch_lat',
    'merch_long': 'merch_long',
    'category': 'category',
    'is_fraud': 'is_fraud',
    'amt': 'amt',
    'unix_time': 'unix_time'
}

# Generate historical features for each column
for col_name, col_key in historical_features.items():
    for i in range(1, N_HISTORICAL + 1):
        df[f'{col_name}_minus_{i}'] = grp[col_key].shift(i)

# Trans time differences for last n transactions (in minutes)
for i in range(1, N_HISTORICAL + 1):
    df[f'trans_time_minus_{i}'] = (
        (df['trans_date_trans_time'] - grp['trans_date_trans_time'].shift(i))
        .dt.total_seconds() / 60
    )


# Average amount for this credit card
df['avg_amt'] = grp['amt'].transform('mean')

# Rolling count of transactions in the last 24 hours
df['trans_count_24h'] = (
    grp.rolling('24H', on='trans_date_trans_time')['amt']
    .count()
    .reset_index(level=0, drop=True)
    .values
)

# labels first
cols = ['trans_date_trans_time', 'is_fraud'] + [c for c in df.columns if c not in ['trans_date_trans_time', 'cc_num', 'trans_num', 'first', 'last', 'is_fraud', 'zip', 'street', 'Unnamed: 0']]
df = df[cols].copy()

# Placeholder for the label encoding
label_encoded = {}

# convert to int64
df['trans_date_trans_time'] = df['trans_date_trans_time'].astype('int64')

string_cols = df.select_dtypes(include=['object']).columns

# Encoding each column that is of type string
for column in string_cols:
    labelEncoder = LabelEncoder()
    df[column] = labelEncoder.fit_transform(df[column].astype(str))

    label_encoded[column] = labelEncoder

# Use pickle to store the label encoding
with open(f"{N_HISTORICAL}_historical_features_encoders.pkl", "wb") as file:
    pickle.dump(label_encoded, file)

# Writing the encoded dataset to the CSV file
df.to_csv(f'{N_HISTORICAL}_historical_features.csv', index=False)
