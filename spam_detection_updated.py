# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    # Try loading the Excel file first
    df = pd.read_excel('emails.xlsx')
except:
    # If Excel file fails, try loading the CSV file
    df = pd.read_csv('mail_data.csv')

# Check the columns in the dataset
print("Dataset columns:", df.columns.tolist())

# If the dataset has different column names, adjust accordingly
# Assuming the message column is named 'message' or 'text' and label column is 'label' or 'spam'
message_column = 'message' if 'message' in df.columns else 'text'
label_column = 'label' if 'label' in df.columns else 'spam'

# Drop any unnecessary columns if they exist
columns_to_drop = [col for col in df.columns if col not in [message_column, label_column]]
if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)

# Convert labels to binary (0 for ham, 1 for spam)
df[label_column] = df[label_column].map({'ham': 0, 'spam': 1})

# Split features and target
X = df[message_column]
y = df[label_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix
    }
    
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
accuracies = [results[model]['accuracy'] for model in models.keys()]
plt.bar(models.keys(), accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot confusion matrices
plt.figure(figsize=(15, 10))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show() 