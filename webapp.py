from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the prediction artifacts
try:
    model = joblib.load('Prediction_Artifacts/final_model_rf.joblib')
    scaler = joblib.load('Prediction_Artifacts/scaler.joblib')
    top_features = joblib.load('Prediction_Artifacts/top_features.joblib')
    all_encoded_columns = joblib.load('Prediction_Artifacts/all_encoded_columns.joblib')
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}")
    # Handle the error appropriately, maybe exit or use default values
    model = None
    scaler = None
    top_features = []
    all_encoded_columns = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded. Please check the artifacts.')

    try:
        # --- 1. Get User Input from Form ---
        input_data = {
            'Age': [int(request.form['Age'])],
            'Income': [float(request.form['Income'])],
            'LoanAmount': [float(request.form['LoanAmount'])],
            'CreditScore': [int(request.form['CreditScore'])],
            'MonthsEmployed': [int(request.form['MonthsEmployed'])],
            'NumCreditLines': [int(request.form['NumCreditLines'])],
            'InterestRate': [float(request.form['InterestRate'])],
            'LoanTerm': [int(request.form['LoanTerm'])],
            'DTIRatio': [float(request.form['DTIRatio'])],
            'Education': [request.form['Education']],
            'EmploymentType': [request.form['EmploymentType']],
            'MaritalStatus': [request.form['MaritalStatus']],
            'HasMortgage': [request.form['HasMortgage']],
            'HasDependents': [request.form['HasDependents']],
            'LoanPurpose': [request.form['LoanPurpose']],
            'HasCoSigner': [request.form['HasCoSigner']]
        }
        
        input_df = pd.DataFrame(input_data)

        # --- 2. Preprocess the Input Data ---
        
        # One-Hot Encode categorical features
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Align columns with the training data
        input_aligned = input_encoded.reindex(columns=all_encoded_columns, fill_value=0)
        
        # Select the top features
        input_top_features = input_aligned[top_features]

        # --- 3. Scale the features ---
        input_scaled = scaler.transform(input_top_features)

        # --- 4. Make Prediction ---
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # --- 5. Display Result ---
        if prediction[0] == 1:
            result_text = 'Prediction: High Risk of Default'
        else:
            result_text = 'Prediction: Low Risk of Default'
            
        probability_text = f"Probability of Default: {prediction_proba[0][1]:.2f}"

        probability_value = prediction_proba[0][1]

        return render_template('index.html', 
                               prediction_text=result_text, 
                               probability_text=probability_text,
                               probability_value=probability_value)

    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
