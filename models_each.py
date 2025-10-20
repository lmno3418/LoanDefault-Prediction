import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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

from sklearn.model_selection import train_test_split

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
from sklearn.ensemble import RandomForestClassifier 

rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

# Fit ONLY on the training data
rf_classifier.fit(X_train_full, Y_train)
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': X_train_full.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# (Optional: Plotting Feature Importances)
import matplotlib.pyplot as plt

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
from sklearn.preprocessing import StandardScaler

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

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
# Resample the *scaled* training data
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)

print(f"\nNew resampled Y_train class distribution:\n{Y_train_resampled.value_counts()}")
print(f"Shape of X_train_resampled: {X_train_resampled.shape}")


print("\n--- Preprocessing Finished ---")
print("Ready for model training.")
print("Use (X_train_resampled, Y_train_resampled) for training.")
print("Use (X_test_scaled, Y_test) for testing.")


#----------# Logistic_Regression #----------#

from sklearn.linear_model import LogisticRegression

# 1. Initialize and train the model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_resampled, Y_train_resampled)

# 2. Make predictions on the TEST set
# IMPORTANT: Use the X_test_scaled (not resampled) and Y_test
Y_pred_lr = lr_model.predict(X_test_scaled)

# 3. Evaluate the model
print("--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_lr):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_lr))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_lr))

# Save the trained model
joblib.dump(lr_model, 'final_model_lr.joblib')




#----------# Random Forest Classifier #----------#

from sklearn.ensemble import RandomForestClassifier

# 1. Initialize and train the model
# You can tune n_estimators (e.g., 100 is a common default)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, Y_train_resampled)

# 2. Make predictions on the TEST set
Y_pred_rf = rf_model.predict(X_test_scaled)

# 3. Evaluate the model
print("--- Random Forest Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_rf):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_rf))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_rf))

#Save the trained model
joblib.dump(rf_model, 'final_model_rf.joblib')



#----------# LightGBM Classifier #----------#

# You may need to install it: pip install lightgbm
import lightgbm as lgb

# 1. Initialize and train the model
lgbm_model = lgb.LGBMClassifier(random_state=42)
lgbm_model.fit(X_train_resampled, Y_train_resampled)

# 2. Make predictions on the TEST set
Y_pred_lgbm = lgbm_model.predict(X_test_scaled)

# 3. Evaluate the model
print("--- LightGBM Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_lgbm):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_lgbm))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_lgbm))

#Save the trained model
joblib.dump(lgbm_model, 'final_model_lgbm.joblib')



#----------# XGBoost Classifier #----------#

# You may need to install it: pip install xgboost
import xgboost as xgb

# 1. Initialize and train the model
# XGBClassifier has many parameters; these are good defaults.
xgb_model = xgb.XGBClassifier(random_state=42, 
                            use_label_encoder=False,  # Recommended to set this to False
                            eval_metric='logloss')    # Common metric for binary classification

xgb_model.fit(X_train_resampled, Y_train_resampled)

# 2. Make predictions on the TEST set
Y_pred_xgb = xgb_model.predict(X_test_scaled)

# 3. Evaluate the model
print("--- XGBoost Results ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_xgb):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred_xgb))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred_xgb))

#Save the trained model
joblib.dump(xgb_model, 'final_model_xgb.joblib')



# ---  Save All Prediction Artifacts ---
print("\n--- Saving artifacts for deployment ---")

# 1. Save the trained model
# joblib.dump(lr_model, 'final_model_lr.joblib')
# joblib.dump(rf_model, 'final_model_rf.joblib')        #-> Most Probable I'll Use This One
# joblib.dump(lgbm_model, 'final_model_lgbm.joblib')
# joblib.dump(xgb_model, 'final_model_xgb.joblib')

# 2. Save the fitted scaler
joblib.dump(scaler, 'scaler.joblib')

# 3. Save the list of top N feature names
joblib.dump(top_features, 'top_features.joblib')

# 4. CRITICAL: Save the list of *all* columns after one-hot encoding
# This is needed to align new data correctly
all_encoded_columns = X_train_full.columns.tolist()
joblib.dump(all_encoded_columns, 'all_encoded_columns.joblib')

print("Artifacts saved successfully.")