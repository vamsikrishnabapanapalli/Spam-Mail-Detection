# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

def transform_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""
    # Convert to lowercase
    text = str(text).lower()
    
    # Tokenize
    text = nltk.word_tokenize(text)
    
    # Remove special characters and keep only alphanumeric words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # Apply stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load the dataset
print("\nAttempting to load dataset...")
try:
    # Try loading the Excel file first
    print("Trying to load emails.xlsx...")
    df = pd.read_excel('emails.xlsx')
    print("Successfully loaded emails.xlsx")
except Exception as e:
    print(f"Failed to load Excel file: {str(e)}")
    try:
        # If Excel file fails, try loading the CSV file
        print("\nTrying to load mail_data.csv...")
        df = pd.read_csv('mail_data.csv')
        print("Successfully loaded mail_data.csv")
    except Exception as e:
        print(f"Failed to load CSV file: {str(e)}")
        raise Exception("Could not find either emails.xlsx or mail_data.csv")

# Print dataset information
print("\nDataset Info:")
print(df.info())
print("\nDataset Head:")
print(df.head())
print("\nDataset columns:", df.columns.tolist())

# If the dataset has different column names, adjust accordingly
# Assuming the message column is named 'message', 'text', 'Message' or 'Text'
# and label column is 'label', 'spam', 'Label' or 'Spam'
message_cols = ['message', 'text', 'Message', 'Text']
label_cols = ['label', 'spam', 'Label', 'Spam']

message_column = next((col for col in message_cols if col in df.columns), None)
label_column = next((col for col in label_cols if col in df.columns), None)

if not message_column or not label_column:
    raise Exception(f"Could not find message and label columns. Available columns: {df.columns.tolist()}")

print(f"\nUsing columns: message='{message_column}', label='{label_column}'")

# Drop any unnecessary columns if they exist
columns_to_drop = [col for col in df.columns if col not in [message_column, label_column]]
if columns_to_drop:
    print(f"\nDropping unnecessary columns: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop, axis=1)

# Handle NaN values
print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nRemoving rows with missing values...")
initial_rows = len(df)
df = df.dropna(subset=[message_column, label_column])
final_rows = len(df)
print(f"Removed {initial_rows - final_rows} rows with missing values")

# Print unique values in label column
print(f"\nUnique values in {label_column} column:", df[label_column].unique())

# Convert labels to binary if they're not already
if df[label_column].dtype != np.int64:
    print("\nConverting labels to binary format...")
    label_mapping = {'ham': 0, 'spam': 1, 'Ham': 0, 'Spam': 1}
    df[label_column] = df[label_column].map(label_mapping)
else:
    print("\nLabels are already in binary format")

# Apply text transformation to the message column
print("\nTransforming text...")
df['transformed_text'] = df[message_column].apply(transform_text)

# Remove any rows where transformed text is empty
initial_rows = len(df)
df = df[df['transformed_text'].str.len() > 0]
final_rows = len(df)
if initial_rows != final_rows:
    print(f"Removed {initial_rows - final_rows} rows with empty transformed text")

# Split features and target
X = df['transformed_text']
y = df[label_column].astype(int)  # Ensure labels are integers

print(f"\nFinal dataset size: {len(df)} rows")
print("Label distribution:")
print(y.value_counts())

if len(df) == 0:
    raise Exception("No valid data remaining after preprocessing!")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
print("\nCreating TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_matrix,
        'model': model
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
plt.figure(figsize=(15, 5))
for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 2, i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Save the best model and vectorizer
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = results[best_model_name]['model']
print(f"\nSaving the best model ({best_model_name}) and vectorizer...")

# Save the vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model and vectorizer saved successfully!") 