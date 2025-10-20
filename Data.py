import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

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