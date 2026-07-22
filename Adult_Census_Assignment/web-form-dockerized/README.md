# Adult Census Income Predictor

A modern, full-stack web application that predicts whether an individual's annual income exceeds $50K based on census demographic data. 

This project was built with a premium "glassmorphism" frontend design and includes a Python Flask backend API, containerized with Docker.

## Features
- **Modern UI/UX**: Built with vanilla HTML/CSS/JS, featuring animated backgrounds, a dark-mode theme, and responsive glassmorphism elements.
- **Python Backend API**: A Flask-based server and API designed to serve a Machine Learning model (e.g., Random Forest) for real-time predictions as well as the static frontend.
- **Dynamic Form Handling**: JavaScript intercepts form submissions, displays smooth loading states, and seamlessly handles API responses.
- **Dockerized**: Fully containerized using Docker and Docker Compose for easy setup and deployment across any cloud platform.

## Project Structure
```
├── app.py               # Flask server and API endpoint
├── templates/
│   └── index.html       # Main frontend form
├── static/
│   ├── style.css        # Premium glassmorphism and animated styling
│   └── script.js        # Form handling and API fetch logic
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose for local testing
└── requirements.txt     # Python dependencies
```

## Deployment (Docker)

This project is fully structured for containerized deployment:
1. Ensure your trained model `best_model_random_forest.pkl` is placed in the root directory.
2. Build and run the container locally using Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. The application will be accessible at `http://localhost:8080`.
4. Deploy the Docker image to any container-based cloud service (e.g., AWS ECS, Render, Google Cloud Run).

## Testing Locally

To test the application locally without Docker, make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
python app.py
```
The application will be accessible at `http://127.0.0.1:5000/`.

## Model Requirements
The provided API template is designed to load a `scikit-learn` model. To ensure accurate predictions, remember to mirror any data preprocessing steps (like `StandardScaler` or one-hot encoding) within `app.py` before passing the data to `model.predict()`.
