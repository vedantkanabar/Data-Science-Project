import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

dFrame = pd.read_csv("combined.csv")

label_encoded = {}

string_cols = dFrame.select_dtypes(include=['object']).columns

for column in string_cols:
    labelEncoder = LabelEncoder()
    dFrame[column] = labelEncoder.fit_transform(dFrame[column].astype(str))

    label_encoded[column] = labelEncoder

with open("encoderList.pkl", "wb") as file:
    pickle.dump(label_encoded, file)

dFrame.to_csv("combined_encoded.csv", index=False)



