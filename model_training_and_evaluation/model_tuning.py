import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import NearMiss

# List of models to be evaluated
models = {
    # "Logistic Regression": LogisticRegression(class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "SVM Linear": LinearSVC(class_weight="balanced"),
    # "SVM RBF": SVC(kernel="rbf", class_weight="balanced"),
    # "Naive Bayes": GaussianNB(),
}

param_dists = {
    "Decision Tree": {
        "max_depth": [None, 20, 30, 40, 50],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5)
    },
    "Random Forest": {
        "n_estimators": randint(100, 500),  
        "max_depth": [None, 20, 30, 40, 50],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "bootstrap": [True, False]         
    },
    "SVM Linear": {
        "C": uniform(0.01, 10)                   
    },
    "AdaBoost": {
        "n_estimators": randint(100, 500),
        "learning_rate": uniform(0.001, 2)    
    },
    "KNN": {
        "n_neighbors": randint(3, 10),
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }
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

# Split the cols based on categorical vs. numerical
categorical_cols = ['merchant', 'category', 'city', 'state', 'job', 'gender']
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

with open ("tuning.txt", "w") as file:
    for model_title, model in models.items():
        file.write(f"Hyperparameter tuning for model: {model_title}\n\n")
        print(f"Tuning model: {model_title}...")

        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dists[model_title],
            n_iter=30,
            cv=5,
            n_jobs=-1,
            scoring="average_precision",
            random_state=35
        )
        random_search.fit(data_train_scaled, label_train_resampled)

        file.write(f"Best parameters: {random_search.best_params_}\n\n")

print("Tuning complete. Results saved to tuning.txt")
