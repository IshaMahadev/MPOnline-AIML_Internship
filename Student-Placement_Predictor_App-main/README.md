# Student Placement Prediction Engine (Docker Containerized)

An end-to-end Machine Learning web application designed to evaluate a student's campus placement probability based on core academic performance indicators, soft skills, technical proficiency, and program dedication. The system trains a Random Forest Classifier on a feature matrix and serves real-time binary classifications through a containerized Flask micro-service.

---

## Student Profile
* **Name:** Isha Mahadev
* **Registration Number:** 23BCE10163
* **Course:** B.Tech Computer Science and Engineering
* **Institution:** VIT Bhopal University

---

## System Architecture & Frameworks
* **Language Core:** Python 3.10
* **Container Layer:** Docker Engine
* **Inference Pipeline:** Scikit-learn, Pandas, NumPy
* **Application Framework:** Flask Microkernel
* **Production Gateway:** Gunicorn WSGI Web Server
* **Model Serialization:** Pickle Binary Format

---

## Project Directory Structure
This deployment relies completely on container virtualization layers to stand up the web app infrastructure natively, eliminating platform-specific routing dependencies like `Procfile` or `runtime.txt`:

```text
Student-Placement-Docker/
│
├── static/
│   └── style.css               # Custom UI corporate layout template
│
├── templates/
│   └── index.html              # Main interactive user submission form
│
├── app.py                      # Production server app kernel & routing logic
├── train.py                    # Feature synthesis execution pipeline and model trainer
├── placement_model.pkl         # Serialized Random Forest Classifier binary
├── requirements.txt            # Plaintext python packaging manifest
├── Dockerfile                  # Main container build rule file
└── .gitignore                  # Excludes python local runtime caches and datasets

```

---

## Feature Matrix Mapping

Inputs submitted via the application web form are processed strictly as high-precision floats and integers, bypassing manual threshold logic or hardcoded scripts to run mathematical classifications inside the Random Forest decision tree layout:

| Input Variable Field | Data Metric Type | Value Bounds / Constraints |
| --- | --- | --- |
| **CGPA** | Continuous Float | Scale: `0.00` to `10.00` |
| **Communication Skills** | Continuous Float | Rating: `0.0` to `10.0` |
| **Resume Score** | Continuous Float | Rating: `0.0` to `10.0` |
| **Coding Score** | Continuous Float | Rating: `0.0` to `10.0` |
| **Placement Attendance** | Continuous Float | Percentage: `0.0%` to `100.0%` |

### Output Target Matrix

* **`0` -> Not Placed**: Features mapped below the trained model classification thresholds.
* **`1` -> Placed**: Features successfully satisfied the predictive boundary conditions.

---

## Local Verification and Testing Instructions

### 1. Traditional Workspace Activation

Initialize local testing by setting up your local conda virtual runtime environment:

```bash
# Create a brand-new Conda environment named 'placement' with Python 3.10
conda create -n placement python=3.10 -y

# Activating your workspace profile
conda activate placement

# Pull dependencies directly into the local scope
pip install -r requirements.txt

# Run the training script to generate the model pkl file
python train.py

# Spin up the development microkernel
python app.py

```

Open `http://127.0.0.1:5000` in your web browser.

### 2. Containerized Application Emulation (Requires Docker Desktop)

To review the immutable app runtime state exactly as it will run inside the cloud clusters, compile the container locally:

```bash
# Compile the container build layers
docker build -t student-placement-app .

# Start up the container container and expose the service gateway port
docker run -p 10000:10000 student-placement-app

```

Open your web browser and navigate to `http://localhost:10000`.

---

## Cloud Cluster Container Deployment

This production repository utilizes immediate cloud hooks linked directly to **Render**:

* The cluster system tracks updates to the repository using Git version history on the `main` branch.
* Render automatically reads the root `Dockerfile` to instantiate a robust Linux virtual environment image (`python:3.10-slim`).
* Package tracking and dependency layers are isolated completely from local file system properties.
* The system launches the production Gunicorn web processes cleanly bound to public gateway interface configurations, running continuous calculations without downtime.
