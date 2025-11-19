import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Reading the combined dataset
dFrame = pd.read_csv("combined.csv")

# Placeholder for the label encoding
label_encoded = {}

# Convert the time to unix epoch time which is int type
dFrame['trans_date_trans_time'] = pd.to_datetime(dFrame['trans_date_trans_time']).astype('int64')
dFrame['dob'] = pd.to_datetime(dFrame['dob']).astype('int64')

string_cols = dFrame.select_dtypes(include=['object']).columns

# Encoding each column that is of type string
for column in string_cols:
    labelEncoder = LabelEncoder()
    dFrame[column] = labelEncoder.fit_transform(dFrame[column].astype(str))

    label_encoded[column] = labelEncoder

# Use pickle to store the label encoding
with open("encoderList.pkl", "wb") as file:
    pickle.dump(label_encoded, file)

# Writing the encoded dataset to the combined_encoded.csv file
dFrame.to_csv("combined_encoded.csv", index=False)
