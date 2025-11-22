import os
import gc
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


def getTrainingSample(resampling_method, X_train, Y_train):

    match resampling_method:

        case "Random Sample":
            # return original data as-is
            return (X_train, Y_train)

        case "Random OverSample":
            sampler = RandomOverSampler(random_state=27)
            X_res, Y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return X_res, Y_res

        case "SMOTE":
            sampler = SMOTE(random_state=27)
            X_res, y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return (X_res, y_res)

        case "ADASYN":
            sampler = ADASYN(random_state=27)
            X_res, y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return (X_res, y_res)

        case "Random Undersample":
            sampler = RandomUnderSampler(random_state=27)
            X_res, Y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return X_res, Y_res

        case "NearMiss-1":
            sampler = NearMiss(version=1)
            X_res, Y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return X_res, Y_res

        case "NearMiss-2":
            sampler = NearMiss(version=2, n_jobs=1)
            X_res, Y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return X_res, Y_res

        case "NearMiss-3":
            # added a neighbours_ver3 value of 7 to increase the pool that is sampled from to prevent a sampling error
            sampler = NearMiss(version=3, n_neighbors_ver3=7)
            X_res, Y_res = sampler.fit_resample(X_train, Y_train)
            del sampler
            return X_res, Y_res

        case _:
            # default return original data
            return (X_train, Y_train)


def getResults(resampling_method, Y_test, Y_pred):

    # Calculate metrics
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    # Get confusion matrix
    cm = confusion_matrix(Y_test, Y_pred, normalize='true')

    tn, fp, fn, tp = cm.ravel()

    # Store result in text file
    with open("results.txt", "a") as f:
        f.write(f"\nClassification Report for {resampling_method}:\n\n")

        f.write(f"Accuracy:  {acc * 100:.5f}%\n")
        f.write(f"Precision: {prec * 100:.5f}%\n")
        f.write(f"Recall:    {rec * 100:.5f}%\n")
        f.write(f"F1 Score:  {f1 * 100:.5f}%\n\n")

        f.write(f"True Positive:  {tp * 100:.5f}%\n")
        f.write(f"True Negative:  {tn * 100:.5f}%\n")
        f.write(f"False Positive: {fp * 100:.5f}%\n")
        f.write(f"False Negative: {fn * 100:.5f}%\n\n")

        f.write(classification_report(Y_test, Y_pred, digits=5))
        f.write("\n")

    # Create and save plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
    disp.plot(cmap='Blues', values_format=".5%")
    disp.ax_.set_xlabel("Predicted Class")
    disp.ax_.set_ylabel("Actual Class")
    plt.title(f"Normalized Confusion Matrix for {resampling_method}")
    plt.savefig(f"Confusion Matrix - {resampling_method}.png")
    plt.clf()
    plt.close()

if __name__ == "__main__":

    combined_encoded_data_csv = "../combined_dataset/combined_encoded.csv"

    if os.path.exists("results.txt"):
        os.remove("results.txt")

    # read the file
    print(f"Reading file ${combined_encoded_data_csv} and generating features.")
    dFrame = pd.read_csv(combined_encoded_data_csv)

    # Split into features and the target variable (is_fraud) and fill in the Nan values
    Features = dFrame.drop("is_fraud", axis=1)
    Is_Fraud = dFrame["is_fraud"]
    feature_names = Features.columns

    # Split the data into testing and training keeping the same class balance in the splits
    Features_train, Features_test, Is_Fraud_train, Is_Fraud_test = train_test_split(
        Features, Is_Fraud, test_size=0.2, random_state=27, stratify=Is_Fraud
    )

    imputer = SimpleImputer(strategy="median")

    Features_train = pd.DataFrame(imputer.fit_transform(Features_train), columns=feature_names)
    Features_test = pd.DataFrame(imputer.transform(Features_test), columns=feature_names)

    resampling_methods = [
        "Random Sample",
        "Random OverSample",
        "SMOTE",
        "ADASYN",
        "Random UnderSample",
        "NearMiss-1",
        # "NearMiss-2",
        "NearMiss-3",
    ]

    print()

    for resampling_method in resampling_methods:

        print(f"Running Random Forest Classifier with training data from {resampling_method}.")

        # Get categorical and numerical columns to prepare for standardization
        (Sampling_features_train, Sampling_Is_Fraud_train) = getTrainingSample(
            resampling_method, Features_train, Is_Fraud_train
        )

        categorical_cols = ['merchant', 'category', 'city', 'state', 'job', 'gender']
        numerical_cols = [col for col in Features_train.columns if col not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', 'passthrough', categorical_cols)
            ]
        )

        # Fit only on the training data (after resampling)
        Sampling_features_train_scaled = pd.DataFrame(
            preprocessor.fit_transform(Sampling_features_train),
            columns=numerical_cols + categorical_cols
        )

        # Apply the same transformation to the test data
        Features_test_scaled = pd.DataFrame(
            preprocessor.transform(Features_test),
            columns=numerical_cols + categorical_cols
        )


        print(f"Got the samples from {resampling_method}")

        # Train the model with the same random state
        model = RandomForestClassifier(random_state=27, n_jobs=-1)
        model.fit(Sampling_features_train_scaled, Sampling_Is_Fraud_train)

        Is_Fraud_pred = model.predict(Features_test_scaled)

        print("Model complete, saving results.")

        # Get the results
        getResults(resampling_method, Is_Fraud_test, Is_Fraud_pred)

        print()

        # --- MEMORY CLEANUP ---
        del Sampling_features_train
        del Sampling_Is_Fraud_train
        del Sampling_features_train_scaled
        del Features_test_scaled
        del model
        del Is_Fraud_pred

        # Clear all matplotlib figures & caches
        plt.clf()
        plt.close('all')

        # Force Python + numpy + sklearn C-extensions to free memory
        gc.collect()
