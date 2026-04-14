# 🌍 ArgusX Air Quality | Predictive MLOps & AI Platform

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Airflow](https://img.shields.io/badge/DataOps-Apache_Airflow-017CEE.svg)
![Llama3](https://img.shields.io/badge/LLM-Llama_3.1_%288B%29-04A77D.svg)
![LangChain](https://img.shields.io/badge/Agent-LangChain%20SQL-gray.svg)

## ArgusX Air Quality 
Es una plataforma predictiva avanzada diseñada para monitorear, predecir y alertar sobre riesgos ambientales en Guadalajara, Jalisco. 

Nuestro objetivo es transicionar del monitoreo reactivo a la **prevención proactiva**. El sistema no solo informa el estado actual de la calidad del aire, sino que predice riesgos futuros y emite recomendaciones de salud accionables en tiempo real, operando bajo el marco de la normativa ambiental mexicana (**NOM-037-STPS-2023**).

---

## ¿A quién va dirigido?

*  **Ciudadanía en General:** Para la toma de decisiones diarias informadas (ej. la viabilidad de salir a correr, ventilar el hogar, o niveles de radiación para uso de protector solar).
*  **Población Vulnerable:** Alertas tempranas para personas con afecciones respiratorias (asma, alergias), indicando ventanas de tiempo críticas donde el uso de mascarillas N95 es altamente recomendado.
*  **Investigadores y Analistas:** Acceso rápido y democratizado a datos históricos mediante un agente conversacional inteligente, eliminando la barrera del lenguaje SQL.

---

##  Arquitectura del Sistema (Los 3 Pilares)

### 1. El Motor de Datos (DataOps & ETL)
Orquestado mediante **Apache Airflow** (Astronomer), garantiza que la base de datos (AWS RDS) se mantenga como la fuente de la verdad con datos siempre frescos. Consta de 3 DAGs principales:
* **Backfill DAG:** Carga masiva de datos históricos desde 2021 hasta la fecha (vía OpenWeather API).
* **Hourly Ingestion DAG:** Consulta e inserción de datos climatológicos cada hora.
* **Retraining Trigger DAG:** Disparo automatizado mensual para iniciar el ciclo MLOps.

### 2. El Motor Predictivo (MLOps Cycle)
Basado en **XGBoost (Extreme Gradient Boosting)**, elegido por su alta eficiencia para modelar series temporales y capturar la no linealidad entre variables meteorológicas (viento, temperatura) y contaminantes.
* **Pipeline Modular:** Compuesto por módulos aislados de Ingesta, Validación (detección de *Data Drift*), Transformación y Entrenamiento.
* **Zero Model Decay:** El reentrenamiento autónomo mensual garantiza que el modelo se adapte a los cambios estacionales.
* **Tracking & Deployment:** Monitoreo de hiperparámetros (Optuna/TPE) vía MLflow, y despliegue automático de artefactos a AWS S3.

### 3. El Agente Inteligente (LLM & RAG/SQL)
Un asistente conversacional que actúa como analista de datos ambientales 24/7.
* **LLM Core:** Llama 3.1 (8B) optimizado mediante la API de **Groq** para latencia ultrabaja.
* **Framework:** Agente SQL de **LangChain** capaz de traducir lenguaje natural a consultas seguras para AWS RDS, incorporando protecciones de tokens y agregaciones forzadas.
* **Búsqueda Web (Tool):** Integración con DuckDuckGo para consultar leyes externas (NOM-2023) o contexto adicional en tiempo real.

---

## Tablero de Datos y Predicciones

La plataforma expone los siguientes puntos críticos:
* **Índice NOM-2023 (Riesgo 1-5):** Unificación algorítmica de múltiples contaminantes (PM2.5, PM10, O3, CO, NO2) en un índice ciudadano comprensible y accionable.
* **Pronóstico a 3 Horas:** Predicción de severidad mediante XGBoost para planificación de actividades al aire libre.
* **Variables Meteorológicas:** Temperatura, humedad y velocidad/dirección del viento (factores de dispersión).
* **Geolocalización:** Mapeo exacto de la toma de datos.

---

##  Estructura del Proyecto

```text
Air_pollution/
├── .github/workflows/          # CI/CD pipelines (GitHub Actions)
├── air_quality/                # Módulo principal de MLOps
│   ├── cloud/                  # Conexiones a AWS RDS
│   ├── components/             # Componentes del pipeline (Ingestion, Transform, Trainer, etc.)
│   ├── constant/               # Variables y rutas constantes
│   ├── entity/                 # Entidades de configuración y artefactos
│   ├── exception/              # Manejo de excepciones personalizadas
│   ├── logging/                # Configuración de logs del sistema
│   ├── pipeline/               # Orquestador del Training Pipeline
│   └── utils/                  # Utilidades generales y métricas de ML
├── chat_agent/                 # Módulo del Asistente LLM (LangChain + Groq)
│   ├── chat_agent.py           # Lógica principal del agente SQL/RAG
│   ├── config.py               # Configuración de LLM y Tools
│   └── database_manager.py     # Gestor de consultas seguras a DB
├── data_schema/                # Esquemas de validación (YAML)
├── final_model/                # Artefactos listos para producción (model.pkl, preprocessor.pkl)
├── templates/                  # Frontend básico / vistas web (index.html)
├── tests/                      # Suite de tests unitarios (pytest)
├── app.py                      # API y Web App (FastAPI / Flask)
├── Dockerfile                  # Contenedorización del proyecto
├── main.py                     # Script de ejecución manual del pipeline MLOps
├── mlflow.db                   # Base de datos local para tracking de experimentos
├── requirements.txt            # Dependencias del proyecto
└── setup.py                    # Script de empaquetado del módulo
```
---

## Despliegue Rápido (Local)

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/ArgusX-dev/AirPollution_Prediction_AI.git](https://github.com/ArgusX-dev/AirPollution_Prediction_AI.git)
   cd AirPollution_Prediction_AI


