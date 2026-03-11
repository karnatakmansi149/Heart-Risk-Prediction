# Heart Disease Prediction Project
# VS Code Ready, Pandas + Sklearn + ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load Dataset
df = pd.read_csv("heart.csv")

# Show first rows
print(df.head())
print("Dataset shape:", df.shape)

# Check for missing values
print(df.isnull().sum())

# Fill missing BMI values with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# 2️⃣ Separate Features and Target
# Predicting heart_disease
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

print("Features after encoding:")
print(X.head())

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5️⃣ Model Prediction
y_pred = model.predict(X_test)

# 6️⃣ Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7️⃣ Heart Disease Prediction Function
def heart_prediction(input_data):
    # Convert input to dataframe with same columns as X
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical columns same as training data
    input_df = pd.get_dummies(input_df)
    
    # Add missing columns if any
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure same column order
    input_df = input_df[X.columns]
    
    # Predict
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        print("Person has Heart Disease Risk")
    else:
        print("Person has No Heart Disease Risk")

# 8️⃣ Example Prediction
example_patient = {
    'id': 9999,
    'age': 55,
    'gender': 'Male',
    'hypertension': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 150,
    'bmi': 28.5,
    'smoking_status': 'formerly smoked',
    'stroke': 0
}

heart_prediction(example_patient)