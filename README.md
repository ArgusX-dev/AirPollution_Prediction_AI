# 🌤️ ArgusX | Air Quality Prediction AI & Analytics
> Plataforma predictiva de calidad del aire propulsada por Machine Learning (XGBoost) y un Agente Autónomo LLM, diseñada para monitorear, predecir y alertar sobre riesgos ambientales en Guadalajara, Jalisco.

## 🎯 ¿Por qué construimos ArgusX?
La contaminación atmosférica es un asesino silencioso en áreas metropolitanas de rápido crecimiento. Construimos ArgusX para resolver un problema de accesibilidad de datos: la información ambiental suele ser estática, difícil de interpretar o llega demasiado tarde. 

Nuestro objetivo fue crear un sistema proactivo que no solo informe el estado actual, sino que **prediga el riesgo a futuro** y proporcione recomendaciones de salud accionables en tiempo real, basándose estrictamente en normativas ambientales mexicanas (NOM-037-STPS-2023 / NOM-172-SEMARNAT-2019).

## 👥 ¿Para quién está dirigido?
* **Ciudadanía General y Deportistas:** Para la toma de decisiones diarias (salir a correr, ventilar la casa, uso de protector solar).
* **Población Vulnerable:** Alertas tempranas para personas con afecciones respiratorias (asma, alergias) indicando cuándo es necesario el uso de mascarillas N95.
* **Investigadores y Analistas:** Acceso rápido a datos históricos mediante nuestro agente interactivo impulsado por IA.

---

## 🧠 Arquitectura y MLOps
Este proyecto es una solución **End-to-End**, abarcando desde la ingesta de datos hasta el despliegue de modelos y la interfaz de usuario.

### 1. El Motor Predictivo (Machine Learning)
* **Modelo:** `XGBoost` (Extreme Gradient Boosting).
* **Por qué lo elegimos:** XGBoost es altamente eficiente para series temporales y datos tabulares complejos, manejando excelentemente la no linealidad entre variables meteorológicas (viento, temperatura, humedad) y la concentración de partículas.
* **Datos:** Entrenado con un dataset histórico masivo (2021-Actualidad) proveniente de OpenWeather API.
* **Pipeline MLOps:** El modelo cuenta con un pipeline de reentrenamiento automatizado mensual para adaptarse a los cambios estacionales e incorporar la data más reciente, evitando el *model drift*.

### 2. El Agente de IA (Argus Assistant)
Hemos integrado un asistente conversacional RAG/SQL que actúa como analista de datos:
* **LLM Core:** Llama 3.1 (8B) optimizado mediante la API de Groq para inferencia ultrarrápida.
* **Framework:** LangChain con herramientas personalizadas (`create_sql_agent`).
* **Routing Dinámico:** El agente es capaz de discernir entre:
  * **Charla natural:** Interacciones y saludos sin coste computacional.
  * **Consultas de Base de Datos:** Generación de SQL seguro para extraer promedios y métricas de AWS RDS, con mecanismos de protección contra desbordamiento de tokens (`top_k` y agregaciones forzadas).
  * **Búsqueda Web:** Integración con DuckDuckGo para consultar leyes ambientales externas (NOM-2023) en tiempo real.

### 3. Frontend y UI/UX
* **Tecnologías:** Vanilla HTML/JS, Tailwind CSS, Leaflet.js (Mapas espaciales) y Chart.js (Series de tiempo).
* **Diseño Orientado a la Acción:** En lugar de solo mostrar concentración de microgramos por metro cúbico (lo cual confunde al usuario promedio), el dashboard traduce los datos de PM2.5 y PM10 a una escala de Riesgo del 1 al 5 (Buena a Extrema), inyectando dinámicamente tarjetas de recomendación sanitaria (ej. "Uso de N95 obligatorio").

---

## 📊 ¿Qué datos mostramos y por qué?
* **Temperatura y Viento:** Variables clave que influyen en la dispersión o estancamiento de los contaminantes.
* **Riesgo NOM-2023 (1-5):** Unificación de múltiples métricas (PM2.5, PM10, O3, CO, NO2) en un índice único y comprensible para el ciudadano.
* **Pronóstico a 3 Horas:** Permite a los usuarios planificar sus actividades al aire libre con anticipación.
* **Geolocalización (Mapa):** Para confirmar visualmente el punto exacto de la toma de datos meteorológicos y analíticos.

---

## 📂 Estructura del Proyecto

```text
ArgusX_AirPollution/
│
├── api/                    # Backend FastAPI (Rutas, Endpoints)
├── chat_agent/             # Lógica del Agente LLM (LangChain, Groq, DuckDuckGo)
├── database/               # Conectores y gestores de AWS RDS
├── ml_pipeline/            # Scripts de ingesta, limpieza y reentrenamiento XGBoost
├── static/                 # Recursos estáticos (Imágenes, CSS custom)
├── templates/              # Interfaz de usuario (index.html, dashboard)
├── config.py               # Variables de entorno y configuración general
├── app.py                  # Entry point del servidor ASGI
└── requirements.txt        # Dependencias de Python
