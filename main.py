# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset_path = 'intents.json'

# Read the JSON file into a Pandas DataFrame
with open(dataset_path, 'r') as file:
    data = pd.read_json(file)
    # Access the nested structure
    data = pd.json_normalize(data['intents'])

# 1. Data Collection
# Assuming columns like 'tag', 'patterns', 'responses'
# Replace column names with your actual column names

# 2. Data Preprocessing
# Handle missing values and outliers
data = data.dropna()  # Remove rows with missing values
# Additional preprocessing steps if needed

# Normalize or standardize features
scaler = StandardScaler()
features = ['tag', 'patterns', 'responses']
data[features] = scaler.fit_transform(data[features])

# 3. Feature Selection
# Use feature selection techniques to identify influential variables
# Example: SelectKBest, Recursive Feature Elimination (RFE), etc.

# 4. Model Development
# Split the dataset into features (X) and target variable (y)
X = data[features]
print(X)

y = data['target_disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize machine learning models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5. Cross-Validation
    cross_val_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores for {model_name}: {cross_val_scores}")

    # 6. Hyperparameter Tuning
    # Example for Random Forest
    param_grid = {'n_estimators': [50, 100, 150],
                  'max_depth': [None, 10, 20, 30]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best hyperparameters for {model_name}: {best_params}")

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Performance metrics for {model_name}:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
    print("----------------------------------------------------")
