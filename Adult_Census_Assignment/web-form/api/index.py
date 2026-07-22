from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the model when the function starts
# Vercel serverless functions load from the root of the api folder or project
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model_random_forest.pkl')

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not found. Please ensure best_model_random_forest.pkl is in the api folder.'}), 500
        
    try:
        data = request.json
        
        # In a real scenario, you need to preprocess the incoming data exactly 
        # as you did during training (e.g., One-Hot Encoding, StandardScaler).
        # For this template, we assume the model can take a DataFrame directly
        # or we manually format the features to match the expected columns.
        
        # Example of creating a DataFrame from single row:
        df = pd.DataFrame([data])
        
        # ⚠️ CRITICAL: You must apply the SAME transformations (get_dummies, StandardScaler) 
        # here before calling model.predict(). This requires loading your fitted scaler 
        # and knowing the exact feature columns of the training set.
        
        # Dummy mock response since real preprocessing code depends on your training notebook
        # prediction = model.predict(X_scaled)[0]
        # probability = model.predict_proba(X_scaled)[0][1]
        
        # Placeholder for successful execution (replace with actual prediction logic)
        score = 0
        if int(data.get('age', 0)) > 30: score += 1
        prediction = 1 if score > 0 else 0
        probability = 0.85
        
        if prediction == 1:
            label = '>50K'
            message = "High likelihood of earning over $50,000 annually."
        else:
            label = '<=50K'
            message = "Predicted annual income is $50,000 or below."
            
        return jsonify({
            'label': label,
            'probability': probability,
            'message': message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
