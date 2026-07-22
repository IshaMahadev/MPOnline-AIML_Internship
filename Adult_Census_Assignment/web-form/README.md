# Adult Census Income Predictor

A modern, full-stack web application that predicts whether an individual's annual income exceeds $50K based on census demographic data. 

This project was built with a premium "glassmorphism" frontend design and includes a serverless Python backend API, ready to be deployed on Vercel.

## 🚀 Features
- **Modern UI/UX**: Built with vanilla HTML/CSS/JS, featuring animated backgrounds, a dark-mode theme, and responsive glassmorphism elements.
- **Python Backend API**: A Flask-based serverless function built for Vercel, designed to serve a Machine Learning model (e.g., Random Forest) for real-time predictions.
- **Dynamic Form Handling**: JavaScript intercepts form submissions, displays smooth loading states, and seamlessly handles API responses.

## 📁 Project Structure
```
├── api/
│   └── index.py         # Flask serverless function for Vercel deployment
├── index.html           # Main frontend form
├── style.css            # Premium glassmorphism and animated styling
├── script.js            # Form handling and API fetch logic
└── requirements.txt     # Python dependencies for the Vercel backend
```

## 🛠️ Deployment (Vercel)

This project is fully structured for instant deployment on [Vercel](https://vercel.com/):
1. Create a new project on Vercel and import this GitHub repository.
2. Ensure your trained model `best_model_random_forest.pkl` is placed inside the `api/` directory (ignored by git due to size limits or privacy, so you might need to upload it or generate it dynamically).
3. Vercel will automatically build the static frontend files and map the `api/index.py` file to the `/api/predict` serverless endpoint.
4. Click **Deploy**!

## 🧪 Testing Locally

To test the frontend locally:
- Simply double-click the `index.html` file to open it in your browser.

To run the Python Flask API locally:
```bash
cd api
pip install -r ../requirements.txt
python index.py
```
*Note: Make sure to adjust your frontend `fetch` URL in `script.js` to point to `http://localhost:5000/api/predict` when testing locally.*

## 📈 Model Requirements
The provided API template is designed to load a `scikit-learn` model. To ensure accurate predictions, remember to mirror any data preprocessing steps (like `StandardScaler` or one-hot encoding) within `api/index.py` before passing the data to `model.predict()`.
