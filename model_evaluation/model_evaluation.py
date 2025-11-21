import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

# List of models to be evaluated
modelsList = {
    "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
    # "SVM RBF": SVC(kernel="rbf", class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier()
}

# Reading the dataset and splitting it into training and testing dataset
dFrame = pd.read_csv("../combined_dataset/combined_encoded.csv")

label = dFrame["is_fraud"]
data = dFrame.drop("is_fraud", axis=1)
feature_names = data.columns

data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.2, stratify=label, random_state=35
)

# Handling NaN value (Filling it with column median) for training and testing data
imputer = SimpleImputer(strategy="median")

data_train = pd.DataFrame(imputer.fit_transform(data_train), columns=feature_names)
data_test = pd.DataFrame(imputer.transform(data_test), columns=feature_names)

# Opening the file to store the evaluation results
with open("results.txt", "w") as file:
    for model_title, model in modelsList.items():
        file.write(f"Evaluation Result for model: {model_title}\n\n")

        print(f"Training model: {model_title}...")

        # Training the model
        model.fit(data_train, label_train)
        label_predict = model.predict(data_test)

        # Taking the performance metric of the model
        precision = precision_score(label_test, label_predict, zero_division=0)
        recall = recall_score(label_test, label_predict, zero_division=0)
        f1 = f1_score(label_test, label_predict, zero_division=0)
        cm = confusion_matrix(label_test, label_predict)

        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # Writing the metric results to the results.txt file
        file.write(f"Accuracy:  {accuracy*100:.5f}%\n")
        file.write(f"Precision: {precision*100:.5f}%\n")
        file.write(f"Recall:    {recall*100:.5f}%\n")
        file.write(f"F1 Score:  {f1*100:.5f}%\n\n")

        file.write(f"True Positive:  {TP / (TP + FN) * 100:.5f}%\n")
        file.write(f"True Negative:  {TN / (TN + FP) * 100:.5f}%\n")
        file.write(f"False Positive: {FP / (FP + TN) * 100:.5f}%\n")
        file.write(f"False Negative: {FN / (FN + TP) * 100:.5f}%\n\n")

        file.write(classification_report(label_test, label_predict, digits=5))
        file.write("\n")

        # Permutation Importance to show the top features
        file.write("Permutation Importance (Top 5 Features):\n")

        r = permutation_importance(model, data_test, label_test, n_repeats=5, random_state=27)

        ranked = sorted(zip(feature_names, r.importances_mean), key=lambda x: x[1], reverse=True)

        for feature, importance_score in ranked[:5]:
                file.write(f"{feature}: {importance_score:.6f}\n")

        file.write("\n")

print("Model evaluation completed. Results saved to results.txt")








