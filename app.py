import os
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv
import boto3
from air_quality.utils.main_utils.utils import load_object
import asyncio
from fastapi import BackgroundTasks


load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

try:
    model = load_object('final_model/model.pkl')
    preprocessor = load_object('final_model/preprocessor.pkl')
    print("Local models uploaded.")
except Exception as e:
    print(f"Warning models are not uploaded, starting service without fresh models: {e}")
    model = None
    preprocessor = None
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "argus_secret")


def download_and_reload_model():
    global model, preprocessor

    print("Starting hot reload of the model...")
    try:
        s3 = boto3.client('s3')
        bucket_name = "airpollutionpredictor"

        s3.download_file(bucket_name, 'final_model/model.pkl', 'final_model/model.pkl')
        s3.download_file(bucket_name, 'final_model/preprocessor.pkl', 'final_model/preprocessor.pkl')

        # 2. Asignamos directo a las variables globales (adiós app.state)
        preprocessor = load_object('final_model/preprocessor.pkl')
        model = load_object('final_model/model.pkl')

        print("Model successfully updated in production!")

    except Exception as e:
        print(f"Error updating the model: {e}")

@app.post("/api/admin/reload-model")
async def reload_model(background_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    if x_api_key != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    background_tasks.add_task(download_and_reload_model)
    return {"message": "Model update started in the background"}

BREAKPOINTS_2023 = {
    'pm10': [(0, 45, 'Good', 1), (45.1, 50, 'Acceptable', 2), (50.1, 132, 'Bad', 3), (132.1, 214, 'Very Bad', 4)],
    'pm2_5': [(0, 25, 'Good', 1), (25.1, 33, 'Acceptable', 2), (33.1, 79, 'Bad', 3), (79.1, 97.4, 'Very Bad', 4)],
    'o3': [(0, 0.058, 'Good', 1), (0.059, 0.090, 'Acceptable', 2), (0.091, 0.135, 'Bad', 3),
           (0.136, 0.175, 'Very Bad', 4)],
    'co': [(0, 8.75, 'Good', 1), (8.76, 11.0, 'Acceptable', 2), (11.1, 13.3, 'Bad', 3), (13.4, 15.5, 'Very Bad', 4)]
}

def get_health_index(concentration, pollutant):
    if pollutant not in BREAKPOINTS_2023: return ('Unknown', 0)
    for (c_low, c_high, category, severity) in BREAKPOINTS_2023[pollutant]:
        if c_low <= concentration <= c_high: return (category, severity)
    return ('Extremely Bad', 5)

def get_real_time_data():
    lat, lon = "20.6597", "-103.3496"

    api_key = os.getenv("OPENWEATHER_API_KEY") or os.getenv("API_KEY")

    if not api_key:
        raise ValueError("API KEY not found! Check your .env file")

    w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    a_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    w_res = requests.get(w_url).json()
    a_res = requests.get(a_url).json()

    if 'list' not in a_res:
        print(f"OPENWEATHER ERROR (Air): {a_res}")
        raise KeyError("The API did not return 'list' for pollution. Check the console.")

    if 'main' not in w_res:
        print(f"OPENWEATHER ERROR (Weather): {w_res}")
        raise KeyError("The API did not return 'main' for weather. Check the console.")

    comp = a_res['list'][0]['components']

    cat_pm25, sev_pm25 = get_health_index(comp["pm2_5"], 'pm2_5')
    cat_pm10, sev_pm10 = get_health_index(comp["pm10"], 'pm10')
    cat_o3, sev_o3 = get_health_index(comp["o3"] / 1960.8, 'o3')
    cat_co, sev_co = get_health_index(comp["co"] / 1145.0, 'co')

    severities = {'pm2_5': sev_pm25, 'pm10': sev_pm10, 'o3': sev_o3, 'co': sev_co}
    main_pollutant = max(severities, key=severities.get)
    max_severity = severities[main_pollutant]

    return {
        'temp': w_res['main']['temp'],
        'hum': w_res['main']['humidity'],
        'press': w_res['main']['pressure'],
        'w_speed': w_res['wind']['speed'],
        'w_dir': w_res['wind']['deg'],
        'clouds': w_res['clouds']['all'],
        'pm2_5': comp['pm2_5'],
        'pm10': comp['pm10'],
        'co': comp['co'],
        'no2': comp['no2'],
        'o3': comp['o3'],
        'current_risk_severity': max_severity,
        'main_pollutant': main_pollutant
    }


@app.post("/api/admin/start-training")
async def start_training_process(x_api_key: str = Header(None)):
    from air_quality.pipeline.training_pipeline import TrainingPipeline
    if x_api_key != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    print("Airflow requested to start retraining. Processing...")
    try:
        pipeline = TrainingPipeline()
        model_trainer_artifact, model_pusher_artifact = pipeline.run_pipeline()
        print("Training completed and uploaded to S3.")
        return {
            "status": "success",
            "message": "Pipeline completed",
            "trainer": str(model_trainer_artifact),
            "pusher": str(model_pusher_artifact)
        }
    except Exception as e:
        print(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/dashboard")
async def get_dashboard_data():

    if preprocessor is None or model is None:
        raise HTTPException(
            status_code=503,
            detail="The AI models are not yet loaded. Run the pipeline in Airflow or wait for synchronization with S3."
        )
    try:
        real_data = get_real_time_data()
        print(f"Weather: {real_data}")

        if not real_data:
            raise ValueError("OpenWeather returned empty data.")

    except Exception as e:
        print(f"Fatal error retrieving weather data: {e}")
        raise HTTPException(status_code=502, detail=f"Failure to connect to the satellite/weather: {e}")

    predictions = []
    last_known_risk = real_data.get('current_risk_severity', 1)

    for h in range(3):
        input_dict = {
            'temperature_c': real_data.get('temp', 0),
            'humidity_pct': real_data.get('hum', 0),
            'pressure_hpa': real_data.get('press', 0),
            'wind_speed_ms': real_data.get('w_speed', 0),
            'wind_direction_deg': real_data.get('w_dir', 0),
            'cloudiness_pct': real_data.get('clouds', 0),
            'co': real_data.get('co', 0),
            'no2': real_data.get('no2', 0),
            'o3': real_data.get('o3', 0),
            'pm2_5': real_data.get('pm2_5', 0),
            'pm10': real_data.get('pm10', 0),
            'hour': (pd.Timestamp.now().hour + h) % 24,
            'day_of_week': pd.Timestamp.now().dayofweek,
            'month': pd.Timestamp.now().month,
            'risk_severity_lag_1h': last_known_risk,
            'risk_severity_lag_2h': last_known_risk,
            'risk_severity_lag_24h': last_known_risk
        }

        df = pd.DataFrame([input_dict])
        X_transform = preprocessor.transform(df)

        raw_pred = model.predict(X_transform)[0]
        final_pred = int(np.clip(np.round(raw_pred), 1, 5))

        predictions.append(final_pred)
        last_known_risk = final_pred

    return {
        "current": real_data,
        "forecast": predictions
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)