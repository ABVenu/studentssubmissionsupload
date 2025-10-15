# 📈 Market-Forecast — ML-based Stock Price Prediction

A robust machine learning system for predicting stock prices using historical market data and technical indicators. This repository provides a full-stack solution with modular backend, frontend, and AI/ML components, enabling research, deployment, and visualization of market forecasts.

## 📊 Features

- **AI/ML Engine:** Advanced models for time-series forecasting and feature engineering in `AI_ML/`.
- **Backend API:** RESTful backend for data access, model serving, and integration under `backend/`.
- **Frontend Dashboard:** Interactive web interface for visualization and analytics located in `frontend/`.
- **Dockerized Deployment:** Seamless setup via `docker-compose.yml` for reproducible environments.
- **Package Management:** Includes both Python (`setup.py`) and Node.js (`package.json`) support.

## 📁 Directory Structure

```
AI_ML/              # Machine learning models, training scripts, experiments
backend/            # Backend server (API, business logic)
frontend/           # Client-side application (dashboard, visualization)
node_modules/       # Node.js dependencies (generated)
.dockerignore       # Docker ignore rules
.gitignore          # Git ignore rules
docker-compose.yml  # Multi-container orchestration
LICENSE             # Project license (open source)
package.json        # Node.js project metadata
package-lock.json   # Node.js exact dependency versions
README.md           # Repository documentation
setup.py            # Python package/setup configuration
```

## 🚀 Getting Started

### ⚙️ Prerequisites

- Python 3.10+

### 📦 Installation & Setup

1. **🌱 Clone the repository:**
   ```bash
   git clone https://github.com/isohamghosh/market-forecast.git
   cd market-forecast
   ```

2. **▶️ Build and run with Docker Compose:**
   ```bash
   python setup.py
   ```

## 📘 Usage

- **Web dashboard:** Visit `http://localhost:3000` (or the configured port) for interactive forecasts.
- **API:** Use backend endpoints for programmatic access to predictions and historical data.
- **Custom modeling:** Extend or update models in `Google Collab` to improve accuracy or add new features. Try to use GPU for faster training.

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1trQxoJwNYRsnZVR1ZNwpokJYOHZlNjxe?usp=sharing)

## 🛠️ Development

   1. **🌱 Clone the repository**

   2. **🖥️ Start Backend Server:**
      ```bash
      cd backend
      uvicorn main:app --reload
      ```

   3. **🎨 Start Frontend Server:**
      ```bash
      cd frontend
      npm start
      ```
   4. **🧠 AI Model:** Recommended to use Google collab file to train model from here - [Google Colab](https://colab.research.google.com/drive/1trQxoJwNYRsnZVR1ZNwpokJYOHZlNjxe?usp=sharing)

   


## 🤝 Contributing

Contributions are welcome! Submit issues or pull requests for bug fixes, feature requests, or enhancements.

## 📫 Contact

For questions or support, open an issue or contact [@isohamghosh](https://github.com/isohamghosh).
