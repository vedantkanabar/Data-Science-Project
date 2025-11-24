from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import os
import numpy as np

# List of models to be evaluated (Taken from the top 3 models from model evaluation)
modelsList = {
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        bootstrap=False,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=353
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=357,
        learning_rate=np.float64(1.7009102740624293)
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=4,
        min_samples_split=9
    ),
}

# List of the dataset to be used
datasetList = {
    1: "1_historical_features.csv", 
    3: "3_historical_features.csv", 
    5: "5_historical_features.csv", 
    7: "7_historical_features.csv"
}

for index, dataset in datasetList.items():
    df = pd.read_csv(f"../historical_features/{dataset}")

    label = df["is_fraud"]
    data = df.drop("is_fraud", axis=1)
    feature_names = data.columns

    data_train, data_test, label_train, label_test = train_test_split(
        data, label, test_size=0.2, stratify=label, random_state=35
    )

    # Handling NaN value (Filling it with column median) for training and testing data
    imputer = SimpleImputer(strategy="median")

    data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=feature_names)
    data_test = pd.DataFrame(imputer.transform(data_test), columns=feature_names)

    # Split the cols based on categorical vs. numerical
    categorical_cols = ['merchant', 'category', 'city', 'state', 'job', 'gender', 'trans_count_24h']
    for i in range(1, index+1):
        categorical_cols.extend([f"merchant_minus_{i}", f"category_minus_{i}", f"is_fraud_minus_{i}"])
    numerical_cols = [col for col in data_train.columns if col not in categorical_cols]

    # Resample the data using NearMiss-1
    nearmiss_sampler = NearMiss(version=1)
    data_train_resampled, label_train_resampled = nearmiss_sampler.fit_resample(data_train, label_train)

    # Initialized the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', 'passthrough', categorical_cols)
        ]
    )

    # Fit only on the resampled training data
    data_train_scaled = pd.DataFrame(
        preprocessor.fit_transform(data_train_resampled),
        columns=numerical_cols + categorical_cols
    )

    # Transform test data
    data_test_scaled = pd.DataFrame(
        preprocessor.transform(data_test),
        columns=numerical_cols + categorical_cols
    )


    folder = f"./{index}_historical_result"
    os.makedirs(folder, exist_ok=True)

    # Opening the file to store the evaluation results
    with open(f"{folder}/results_{index}_historical.txt", "w") as file:
        for model_title, model in modelsList.items():
            file.write(f"Evaluation Result for model: {model_title}\n\n")

            # Training the model
            model.fit(data_train_scaled, label_train_resampled)
            label_predict = model.predict(data_test_scaled)

             # Taking the performance metric of the model
            precision = precision_score(label_test, label_predict)
            recall = recall_score(label_test, label_predict)
            f1 = f1_score(label_test, label_predict)
            cm = confusion_matrix(label_test, label_predict, normalize='true')

            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)

             # Writing the metric results to the results.txt file
            file.write(f"Accuracy:  {accuracy*100:.5f}%\n")
            file.write(f"Precision: {precision*100:.5f}%\n")
            file.write(f"Recall:    {recall*100:.5f}%\n")
            file.write(f"F1 Score:  {f1*100:.5f}%\n\n")

            file.write(f"True Positive:  {TP * 100:.5f}%\n")
            file.write(f"True Negative:  {TN * 100:.5f}%\n")
            file.write(f"False Positive: {FP * 100:.5f}%\n")
            file.write(f"False Negative: {FN * 100:.5f}%\n\n")

            file.write(classification_report(label_test, label_predict, digits=5))
            file.write("\n")

            # Permutation Importance to show the top features
            file.write("Permutation Importance (Top 10 Features):\n")

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
            disp.plot(cmap='Blues', values_format=".5%")
            disp.ax_.set_xlabel("Predicted Class")
            disp.ax_.set_ylabel("Actual Class")
            plt.title(f"Confusion Matrix for {model_title}\n({index} Historical Transactions)") 
            plt.savefig(f"{folder}/Confusion Matrix - {model_title} - {index} Historical Features.png")
            plt.clf()
            plt.close()

            r = permutation_importance(model, data_test_scaled, label_test, n_repeats=5, random_state=27)

            ranked = sorted(zip(feature_names, r.importances_mean), key=lambda x: x[1], reverse=True)

            for feature, importance_score in ranked[:10]:
                file.write(f"{feature}: {importance_score:.6f}\n")
        
            top_features = ranked[:10]
            features = [f for f, _ in top_features]
            scores = [s for _, s in top_features]

            # Plot for feature importance
            plt.figure(figsize=(8, 5))
            plt.barh(features, scores)
            plt.xlabel("Importance Score (Mean Decrease in Accuracy)")
            plt.title("Top 10 Features")
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Save the figure
            plt.savefig(f"{folder}/Feature_Importance_{model_title.replace(' ', '_')}_{index}_Historical.png")

            # Clear the figure to free memory
            plt.clf()
            plt.close()

            file.write("\n")
    
    print(f"Evaluation with {index} historical data completed.")