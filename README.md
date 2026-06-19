# 🚀 GPU Cloud Resource Forecasting

Welcome to the **GPU Cloud Resource Forecasting** project! 🎉 This repository contains an end-to-end Machine Learning ecosystem designed to predict and manage GPU compute resources in a cloud environment. 

---

## 🧐 What is the Problem?
In modern cloud computing platforms, GPUs are highly sought after but incredibly expensive. 
- **Under-provisioning** leads to slow-downs, rejected jobs, and unhappy users. 📉
- **Over-provisioning** leads to idle GPUs and massive financial waste. 💸

Cloud providers and large organizations struggle to predict **when** and **how much** GPU capacity they will need.

## 💡 What is this Project?
This project is an **End-to-End Machine Learning Pipeline** and **Monitoring System** that analyzes past GPU usage data to **forecast future demand**. 

### ✨ What it Does:
1. **Ingests** raw GPU usage data.
2. **Trains** state-of-the-art ML models (like XGBoost, PyTorch) to predict future resource needs.
3. **Tracks** model performance and versions using MLflow.
4. **Serves** these predictions via a FastAPI backend.
5. **Visualizes** everything on a beautiful Grafana dashboard with real-time Prometheus metrics.

### ⚙️ How it Works:
1. **Data Scientists** use Jupyter Notebooks (`notebooks/`) to experiment with data and train models.
2. Models are saved to `outputs/models/` and registered in **MLflow**.
3. The **FastAPI Backend** (`backend/`) loads the trained model and exposes REST endpoints.
4. **Prometheus** scrapes system and backend metrics.
5. **Grafana** reads from Prometheus and the Backend to display live forecasting dashboards to stakeholders! 📊

---

## 🛠️ Tech Stack Explained
This project uses a modern, robust, and scalable tech stack:

### 🧠 Machine Learning & Data Science
* **Python**: The core language for all ML and data processing. 🐍
* **Pandas / NumPy**: For heavy data wrangling and numerical operations. 🐼
* **PyTorch & XGBoost**: The heavy lifters for creating deep learning and tree-based forecasting models. 🔥
* **Jupyter Notebooks**: For interactive data exploration and training. 📓

### 🚀 Backend & API
* **FastAPI**: A blazing fast Python framework used to build our prediction API. ⚡
* **Uvicorn**: The ASGI web server that runs FastAPI.

### 📈 MLOps & Observability
* **MLflow**: Tracks our machine learning experiments, parameters, and saves model versions. 📦
* **Prometheus**: A time-series database that monitors the health and metrics of our API. ⏱️
* **Grafana**: The visualization layer that connects to our API and Prometheus to show gorgeous, real-time dashboards. 🎨

### 🐳 Infrastructure
* **Docker & Docker Compose**: Containerizes all our apps (Backend, MLflow, Grafana, Prometheus) so they run flawlessly on *any* machine with a single command! 🚢

---

## 📂 In-Depth Folder Structure
Here is how the project is organized to keep things clean and scalable:

* `data/` 📁
  * `raw/` - Raw, read-only CSV datasets.
  * `interim/` - Intermediate processed data (parquets, etc).
  * `processed/` - Final Train/Test/Validation datasets ready for ML.
* `notebooks/` 📓 - Numbered Jupyter notebooks (e.g., `01_data_exploration.ipynb`).
* `outputs/` 📤
  * `models/` - Saved model weights `.pkl` or `.pt`.
  * `reports/` - Generated plots and thesis-ready figures.
* `backend/` ⚙️ - The FastAPI source code, Dockerfile, and services.
* `grafana/` & `prometheus/` 📊 - Configuration files and pre-built dashboards.

---

## 🚀 Step-by-Step Setup Guide

Ready to get it running? Follow these simple steps! 🏃‍♂️

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd Cloud
```

### Step 2: Set up the Python Environment (For Model Training)
If you want to train models locally, set up a virtual environment:
```bash
python -m venv gpu_env
source gpu_env/bin/activate  # On Windows use: gpu_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Put Data in Place
Make sure your raw datasets are placed inside the `data/raw/` directory before running notebooks.

---

## 🎮 How to Run the Application

We use Docker Compose to start the entire infrastructure (Backend, MLflow, Grafana, Prometheus) in one click!

### Start Everything:
```bash
docker-compose up --build -d
```
*(The `-d` runs it in the background so your terminal stays free!)*

### Access the Services 🌐:
Once running, you can access the different components in your browser:
- **FastAPI Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) (Default dashboard is pre-loaded!)
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090)

### Stop Everything:
When you are done, gracefully shut it down:
```bash
docker-compose down
```

---

## 🛠️ How to Fix Problems (Troubleshooting)

Having trouble? Here are the most common fixes! 🚑

**1. Port Conflicts (e.g., Port 8000 is already in use)**
* **Problem**: Another app is running on port 8000, 3000, or 5000.
* **Fix**: Change the port mapping in `docker-compose.yml`. For example, change `8000:8000` to `8001:8000` for the backend.

**2. Docker Containers Won't Start**
* **Problem**: Error saying "Docker daemon is not running".
* **Fix**: Make sure Docker Desktop is open and running on your machine before running `docker-compose up`.

**3. MLflow Database Error**
* **Problem**: Corrupted `mlflow.db`.
* **Fix**: Delete the `mlflow.db` file and the `mlruns/` directory (if you don't mind losing past experiment logs), then restart docker-compose. It will create a fresh one!

**4. Grafana Dashboards are Empty**
* **Problem**: No data showing in Grafana.
* **Fix**: Check if your FastAPI backend is running properly. Ensure you have trained a model and saved the outputs so the backend has data to serve. You can check the backend logs via:
  ```bash
  docker logs gpu-forecast-api
  ```

**5. Missing Python Packages (Local Training)**
* **Problem**: `ModuleNotFoundError` when running Jupyter notebooks.
* **Fix**: Ensure your virtual environment is activated (`gpu_env\Scripts\activate`) and you ran `pip install -r requirements.txt`. Install any missing package manually with `pip install <package-name>`.

---
*Built with ❤️ for optimizing Cloud GPU Resources.*
