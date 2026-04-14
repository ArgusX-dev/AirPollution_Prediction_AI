import sys
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from air_quality.exception.exception import AirQualityException
from langchain.tools import tool
from .config import Config
from app import model,preprocessor,get_real_time_data
import numpy as np
import pandas as pd


@tool
def predict_future_air_quality(hours_ahead: int = 3) -> str:

    from app import model, preprocessor, get_real_time_data

    if model is None or preprocessor is None:
        return "El modelo predictivo no esta cargado. No puedo calcular el futuro en este momento."

    try:
        real_data = get_real_time_data()
        last_known_risk = real_data.get('current_risk_severity', 1)
        final_pred = last_known_risk

        for h in range(hours_ahead):
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

            last_known_risk = final_pred

        mapa_riesgos = {
            1: "Nivel 1 (Buena)",
            2: "Nivel 2 (Aceptable)",
            3: "Nivel 3 (Mala)",
            4: "Nivel 4 (Muy Mala)",
            5: "Nivel 5 (Extremadamente Mala)"
        }
        riesgo_texto = mapa_riesgos.get(final_pred, "Desconocido")

        return f"El pronostico para dentro de {hours_ahead} hora(s) indica una calidad de aire de {riesgo_texto}."

    except Exception as e:
        return f"Hubo un error interno al consultar el motor XGBoost: {str(e)}"


class SQLAgentBuilder:
    def __init__(self, db):
        self.db = db
        self.model = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.RAG_MODEL,
            temperature=0.3
        )
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()

        current_history = self.history_store[session_id]

        if len(current_history.messages) > 6:
            current_history.messages = current_history.messages[-6:]

        return current_history

    def build_agent(self):
        try:
            web_search_tool = DuckDuckGoSearchRun(
                name="internet_search",
                description="Useful for searching the internet for general information, laws, regulations (such as NOM-2023), or news that are not present in the SQL database."
            )
            system_prefix = """Eres Argus, el analista de datos de IA para ArgusX. HABLA SIEMPRE EN ESPANOL.

                                    REGLAS DE ORO (SI ROMPES ESTO, EL SISTEMA FALLARA):
                                    1. LIMITE DE 7 DIAS: Cuando pidan datos sin especificar fecha, asume que solo quiere los ultimos 7 dias. NUNCA busques en la base de datos completa.
                                    2. NUNCA MENCIONES LA BASE DE DATOS: Jamas digas "en la tabla" o "la base de datos SQL". Di "Nuestros registros indican".
                                    3. REDONDEO HUMANO: Redondea a MAXIMO un decimal (ej. 21.5).
                                    4. TUS HERRAMIENTAS (USO ESTRICTO): 
                                       - Usa SQL SOLO para datos pasados o actuales.
                                       - Usa 'predict_future_air_quality' SOLO cuando el usuario pregunte por el PRONOSTICO, el FUTURO, o las PROXIMAS HORAS.
                                       - Usa INTERNET SOLO para buscar definiciones o normativas (ej. "NOM-2023 ambiental").
                                    5. RECOMENDACIONES DE SALUD: Al dar un nivel de calidad, da una breve recomendacion (Ej. Nivel 1 o 2: disfruta el aire libre. Nivel 3+: reduce actividades).

                                    Tu objetivo es dar una respuesta natural y directa.
            """
            agent_executor = create_sql_agent(
                llm=self.model,
                db=self.db,
                agent_type="tool-calling",
                extra_tools=[web_search_tool, predict_future_air_quality],
                verbose=True,
                prefix=system_prefix,
                top_k=5
            )

            agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                self._get_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )

            return agent_with_chat_history

        except Exception as e:
            raise AirQualityException(e, sys)