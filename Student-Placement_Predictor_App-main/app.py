from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# loading the classification model binary file
with open('placement_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # parsing form fields to correct primitive datatypes
            cgpa = float(request.form['cgpa'])
            communication_skills = float(request.form['communication_skills'])
            resume_score = float(request.form['resume_score'])
            coding_score = float(request.form['coding_score'])
            attendance_placement = float(request.form['attendance_placement'])

            # packaging parameters into 2d array for scikit-learn inference
            features = np.array([[cgpa, communication_skills, resume_score, coding_score, attendance_placement]])
            
            # running true ml inference
            prediction = model.predict(features)[0]
            
            # mapping prediction bits directly to string outcomes
            if prediction == 1:
                status = "Placed 🎉"
            else:
                status = "Not Placed 😔"
                
            return render_template('index.html', prediction_text=f'Placement Verdict: {status}')
            
        except Exception as e:
            return render_template('index.html', prediction_text="Error updating calculation arrays.")

if __name__ == '__main__':
    app.run(debug=True)