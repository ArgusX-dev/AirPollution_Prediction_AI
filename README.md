# 🌍 ArgusX Air Quality | Predictive MLOps & AI Platform

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Airflow](https://img.shields.io/badge/DataOps-Apache_Airflow-017CEE.svg)
![Llama3](https://img.shields.io/badge/LLM-Llama_3.1_%288B%29-04A77D.svg)
![LangChain](https://img.shields.io/badge/Agent-LangChain%20SQL-gray.svg)

## ArgusX Air Quality 
It is an advanced predictive platform designed to monitor, forecast, and issue alerts about environmental risks in Guadalajara, Jalisco.
Our goal is to transition from reactive monitoring to proactive prevention. The system not only reports the current state of air quality, but also predicts future risks and provides actionable health recommendations in real time, operating under the framework of Mexican environmental regulations (NOM-037-STPS-2023).
---

## Who is it for?

*  **General Public:** For making informed daily decisions (e.g., whether it’s safe to go for a run, ventilate the home, or assess radiation levels for sunscreen use).
*  **Vulnerable Populations:** Early warnings for people with respiratory conditions (asthma, allergies), indicating critical time windows where the use of N95 masks is highly recommended.
*  **Researchers and Analysts::** Fast, democratized access to historical data through an intelligent conversational agent, removing the SQL language barrier.

---

##  System Architecture (The 3 Pillars)

### 1. Data Engine (DataOps & ETL)
Orchestrated using Apache Airflow (Astronomer), it ensures that the database (AWS RDS) remains the single source of truth with consistently fresh data. It consists of 3 main DAGs:
* Backfill DAG: Bulk loading of historical data from 2021 to the present (via OpenWeather API).
* Hourly Ingestion DAG: Querying and inserting weather data every hour.
* Retraining Trigger DAG: Automated monthly trigger to initiate the MLOps cycle.

### 2. Predictive Engine (MLOps Cycle)
Based on XGBoost (Extreme Gradient Boosting), chosen for its high efficiency in modeling time series and capturing non-linearity between meteorological variables (wind, temperature) and pollutants.* **Pipeline Modular:** Compuesto por módulos aislados de Ingesta, Validación (detección de *Data Drift*), Transformación y Entrenamiento.
* Modular Pipeline: Composed of isolated modules for Ingestion, Validation (data drift detection), Transformation, and Training.
* Zero Model Decay: Monthly autonomous retraining ensures the model adapts to seasonal changes.
Tracking & Deployment: Hyperparameter monitoring (Optuna/TPE) via MLflow, and automatic deployment of artifacts to AWS S3.
### 3. Intelligent Agent (LLM & RAG/SQL)
A conversational assistant acting as a 24/7 environmental data analyst.
* LLM Core: Llama 3.1 (8B), optimized via the Groq API for ultra-low latency.
* Framework: LangChain SQL agent capable of translating natural language into secure queries for AWS RDS, incorporating token protections and enforced aggregations.
* Web Search (Tool): Integration with DuckDuckGo to consult external regulations (NOM-2023) or additional real-time context.
---

## Data & Prediction Dashboard

The platform provides the following key outputs:
* NOM-2023 Index (Risk 1–5): Algorithmic unification of multiple pollutants (PM2.5, PM10, O3, CO, NO2) into a citizen-friendly, actionable index.
* 3-Hour Forecast: Severity prediction using XGBoost for planning outdoor activities.
* Meteorological Variables: Temperature, humidity, and wind speed/direction (dispersion factors).
Geolocation: Precise mapping of data collection points.

---

##  Project Structure

```text
Air_pollution/
├── .github/workflows/          # CI/CD pipelines (GitHub Actions)
├── air_quality/                # Main MLOps module
│   ├── cloud/                 # AWS RDS connections
│   ├── components/            # Pipeline components (Ingestion, Transform, Trainer, etc.)
│   ├── constant/              # Constant variables and paths
│   ├── entity/                # Configuration and artifact entities
│   ├── exception/             # Custom exception handling
│   ├── logging/               # System logging configuration
│   ├── pipeline/              # Training pipeline orchestrator
│   └── utils/                 # General utilities and ML metrics
├── chat_agent/                # LLM Assistant module (LangChain + Groq)
│   ├── chat_agent.py          # Main SQL/RAG agent logic
│   ├── config.py              # LLM and Tools configuration
│   └── database_manager.py    # Secure database query manager
├── data_schema/               # Validation schemas (YAML)
├── final_model/               # Production-ready artifacts (model.pkl, preprocessor.pkl)
├── templates/                 # Basic frontend / web views (index.html)
├── tests/                     # Unit test suite (pytest)
├── app.py                     # API and Web App (FastAPI / Flask)
├── Dockerfile                 # Project containerization
├── main.py                    # Manual execution script for the MLOps pipeline
├── mlflow.db                  # Local database for experiment tracking
├── requirements.txt           # Project dependencies
└── setup.py                   # Module packaging script
```
---

## Local Development 

1. **Clone Repository:**
   ```bash
   git clone [https://github.com/ArgusX-dev/AirPollution_Prediction_AI.git](https://github.com/ArgusX-dev/AirPollution_Prediction_AI.git)
   cd AirPollution_Prediction_AI


