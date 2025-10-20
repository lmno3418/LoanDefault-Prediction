import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

# Import all classifiers for comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Set pandas display option
pd.set_option('display.max_columns', None)

# --- 1. Load and Inspect the Data ---
try:
    data = pd.read_csv('./dataset/Loan_default.csv')
except FileNotFoundError:
    print("Error: 'Loan_default.csv' not found. Make sure the file is in the correct directory.")
    exit()

print("--- Data Loaded Successfully ---")

# --- 2. Preprocessing: Column Cleaning & Encoding ---
df = data.copy()

# Correcting column names
df.columns = df.columns.str.strip()

# Split the Data into Features and Target Variable
X = df.drop(['Default', 'LoanID'], axis=1)  # ID is irrelevant to prediction
Y = df['Default']

# Define categorical columns
categorical_columns = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

# Perform one-hot encoding on the full feature set
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
print(f"Original encoded features shape: {X_encoded.shape}")


# --- 3. Split Data into Training and Testing Sets ---
# **CRITICAL STEP**: Split *before* any feature selection or SMOTE
# This prevents data leakage. We split the *full* encoded dataset.

X_train_full, X_test_full, Y_train, Y_test = train_test_split(X_encoded, Y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            stratify=Y) # Stratify handles imbalance during split

print("\n--- Data Splitting Complete (using all features) ---")
print(f"X_train_full (training features) shape: {X_train_full.shape}")
print(f"Y_train (training target) shape:      {Y_train.shape}")
print(f"X_test_full (testing features) shape:   {X_test_full.shape}")
print(f"Y_test (testing target) shape:        {Y_test.shape}")


# --- 4. Feature Importance Analysis ---
# Now, we find feature importance using *only* the training data.
print("\n--- Calculating Feature Importance (on Training Data only) ---")

# Use RandomForestClassifier for a classification problem
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

# Fit ONLY on the training data
rf_classifier.fit(X_train_full, Y_train)
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X_train_full.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# (Optional: Plotting Feature Importances)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='maroon')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances (from Training Data)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- 5. Feature Selection ---
# Dynamically select the top N features based on the importance calculated
N_FEATURES = 9
top_features = importance_df['Feature'].head(N_FEATURES).tolist()

print(f"\n--- Selected Top {N_FEATURES} Features ---")
print(top_features)

# Filter both training and testing sets to keep only these features
X_train = X_train_full[top_features]
X_test = X_test_full[top_features]

print(f"X_train (selected features) shape: {X_train.shape}")
print(f"X_test (selected features) shape:  {X_test.shape}")


# --- 6. Feature Scaling (Performed *before* SMOTE) ---
print("\n--- Scaling Features ---")

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler ONLY on the *original* feature-selected training data (X_train)
X_train_scaled = scaler.fit_transform(X_train)

# Use the SAME fitted scaler to transform the test data (X_test)
X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")
print(f"X_train_scaled (original) shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape:             {X_test_scaled.shape}")


# --- 7. Handle Class Imbalance with SMOTE (Performed *after* Scaling) ---
# Apply SMOTE *only* to the scaled training data
print(f"\nOriginal Y_train class distribution:\n{Y_train.value_counts()}")

smote = SMOTE(random_state=42)
# Resample the *scaled* training data
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)

print(f"\nNew resampled Y_train class distribution:\n{Y_train_resampled.value_counts()}")
print(f"Shape of X_train_resampled: {X_train_resampled.shape}")


print("\n--- Preprocessing Finished ---")
print("Ready for model training.")
print("Use (X_train_resampled, Y_train_resampled) for training.")
print("Use (X_test_scaled, Y_test) for testing.")


# --- 8. Model Comparison using Cross-Validation ---
# This new section uses the logic you provided, applied to the
# resampled training data (X_train_resampled, Y_train_resampled)

print("\n--- Starting Model Comparison (Cross-Validation on Training Data) ---")

classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42, probability=True),  # Enable probability estimates
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'XGBoost': xgb.XGBClassifier(n_estimators=10, random_state=42),
}

# Initialize dictionaries to store evaluation metric results
results = {
    'Classifier': [],
    'Accuracy': [],
    'F1 Score': [],
    'Precision': [],
    'Recall': [],
    'ROC AUC': []
}

# Loop through classifiers and calculate various evaluation metrics using cross-validation
# We use X_train_resampled and Y_train_resampled as the data for this evaluation
for classifier_name, classifier in classifiers.items():
    print(f"Evaluating {classifier_name}...")
    
    # Get probability scores for ROC curve
    # Note: This is run on the *entire* resampled set for plotting,
    # but the metrics below are from proper cross-validation.
    y_scores = cross_val_predict(classifier, X_train_resampled, Y_train_resampled, cv=5, method='predict_proba')[:, 1]
    
    # Get cross-validated metric scores
    accuracy_scores = cross_val_score(classifier, X_train_resampled, Y_train_resampled, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(classifier, X_train_resampled, Y_train_resampled, cv=5, scoring='f1')
    precision_scores = cross_val_score(classifier, X_train_resampled, Y_train_resampled, cv=5, scoring='precision')
    recall_scores = cross_val_score(classifier, X_train_resampled, Y_train_resampled, cv=5, scoring='recall')
    roc_auc_scores = cross_val_score(classifier, X_train_resampled, Y_train_resampled, cv=5, scoring='roc_auc')
    
    # Take the mean of cross-validation scores
    accuracy_mean = np.mean(accuracy_scores)
    f1_mean = np.mean(f1_scores)
    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)
    roc_auc_mean = np.mean(roc_auc_scores)
    
    results['Classifier'].append(classifier_name)
    results['Accuracy'].append(accuracy_mean)
    results['F1 Score'].append(f1_mean)
    results['Precision'].append(precision_mean)
    results['Recall'].append(recall_mean)
    results['ROC AUC'].append(roc_auc_mean)
    
    # ROC curve calculation using the predicted scores
    fpr, tpr, _ = roc_curve(Y_train_resampled, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{classifier_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Create line plots to compare evaluation metrics
plt.figure(figsize=(12, 6))
for metric_name, metric_results in {
    'Accuracy': results['Accuracy'],
    'F1 Score': results['F1 Score'],
    'Precision': results['Precision'],
    'Recall': results['Recall'],
    'ROC AUC': results['ROC AUC']}.items():
    plt.plot(results['Classifier'], metric_results, label=metric_name, marker='o')

plt.xlabel('Classifiers')
plt.ylabel('Score')
plt.title('Classifier Comparison (Cross-Validation)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Print the results as a DataFrame
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Results ---")
print(results_df)