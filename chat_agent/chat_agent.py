import sys
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from air_quality.exception.exception import AirQualityException
from .config import Config

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
            system_prefix = """Eres Argus, el analista de datos de IA para ArgusX. HABLA SIEMPRE EN ESPAÑOL.

                        REGLAS DE ORO (SI ROMPES ESTO, EL SISTEMA FALLARÁ):
                        1. LÍMITE DE 7 DÍAS: Cuando el usuario pida datos sin especificar fecha, asume que solo quiere los últimos 7 días. NUNCA busques en toda la base de datos completa.
                        2. NUNCA MENCIONES LA BASE DE DATOS: Jamás digas "en la tabla weather_pollution" o "la base de datos SQL". Di "Nuestros registros indican" o "El sistema detecta".
                        3. REDONDEO HUMANO: NUNCA des números largos (ej. 21.4825...). Redondea a MÁXIMO un decimal (ej. 21.5).
                        4. TUS HERRAMIENTAS: 
                           - Usa SQL SOLO para métricas, clima, PM10, PM2.5, etc.
                           - Usa INTERNET SOLO para buscar definiciones (ej. "qué es PM2.5") o normativas (agrega siempre la palabra "ambiental" a tu búsqueda, ej. "NOM-2023 ambiental").

                        Tu objetivo es dar una respuesta natural, conversacional y directa, como un colega humano consultando un dashboard.
"""
            agent_executor = create_sql_agent(
                llm=self.model,
                db=self.db,
                agent_type="tool-calling",
                extra_tools=[web_search_tool],
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