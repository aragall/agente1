import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
import datetime
import requests
import xml.etree.ElementTree as ET

# --- Configuración de la página ---
st.set_page_config(page_title="Agente de Tiempo y Clima", page_icon="☀️", layout="wide")

st.title("☀️ Agente Meteorológico ☁️")
st.markdown("""
Este agente te dice la hora, el clima (Open-Meteo) y **alertas oficiales de AEMET**.
""")

# --- Sidebar para configuración ---
with st.sidebar:
    st.header("Configuración")
    
    # Input para la API Key de Google
    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key")
    st.markdown("[Consigue tu Google API Key](https://aistudio.google.com/app/apikey)")
    
    # Input AEMET
    aemet_api_key = st.text_input("AEMET API Key (Opcional)", type="password", key="aemet_api_key")
    st.markdown("[Consigue tu AEMET API Key](https://opendata.aemet.es/centrodescargas/inicio)")

    # Modelos actualizados
    model_options = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
    selected_model = st.selectbox("Modelo", model_options, index=0)
    
    st.divider()
    st.markdown("### Acerca de")
    st.markdown("Creado con LangChain, Streamlit, Open-Meteo y AEMET.")

if not google_api_key:
    st.info("👋 Por favor, ingresa tu **Google API Key** en la barra lateral para comenzar.")
    st.stop()

# --- Herramientas ---

def get_current_time(query: str = "") -> str:
    """Devuelve la fecha y hora actual exacta."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    """
    Obtiene el clima actual y pronóstico para una ubicación dada usando Open-Meteo.
    """
    try:
        # 1. Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=es&format=json"
        geo_res = requests.get(geo_url).json()
        
        if not geo_res.get("results"):
            return f"No encontré la ubicación '{location}'. Intenta ser más específico."
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        name = geo_res["results"][0]["name"]
        
        # 2. Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto"
        weather_res = requests.get(weather_url).json()
        
        current = weather_res.get("current", {})
        daily = weather_res.get("daily", {})
        
        temp = current.get("temperature_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        
        report = f"Clima actual en {name}:\n"
        report += f"- Temperatura: {temp}°C\n"
        report += f"- Viento: {wind} km/h\n"
        
        if daily:
             max_temp = daily.get('temperature_2m_max', ['N/A'])[0]
             min_temp = daily.get('temperature_2m_min', ['N/A'])[0]
             report += f"- Máxima hoy: {max_temp}°C\n"
             report += f"- Mínima hoy: {min_temp}°C\n"
             
        return report
        
    except Exception as e:
        return f"Error al obtener el clima: {str(e)}"

def check_aemet_alerts(location: str) -> str:
    """
    Consulta alertas meteorológicas vigentes en AEMET para una ubicación (provincia o zona).
    Requiere AEMET API Key.
    """
    if not aemet_api_key:
        return "No puedo consultar alertas sin una AEMET API Key configurada."
        
    try:
        # Endpoint para avisos de hoy
        url = "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/archivo/hoy"
        headers = {"api_key": aemet_api_key}
        
        # 1. Obtener URL de datos
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
             return f"Error conectando con AEMET: {res.status_code} - {res.text}"
             
        json_res = res.json()
        if json_res.get("estado") != 200:
             return f"Error API AEMET: {json_res.get('descripcion')}"
             
        data_url = json_res.get("datos")
        
        # 2. Descargar datos (suelen ser XML/JSON)
        # Nota: AEMET a veces devuelve XML en este endpoint aunque pidas JSON en la metadata
        data_res = requests.get(data_url)
        
        # Como es complejo parsear todo el CAP (Common Alerting Protocol), 
        # haremos una búsqueda de texto simple primero para ver si la ubicación está mencionada.
        # En una app real, usaríamos una librería CAP parser.
        
        content = data_res.text
        
        # Búsqueda muy básica para demostración
        # Buscamos si el nombre de la ubicación aparece en las alertas
        if location.lower() in content.lower():
             # Extraer un fragmento alrededor
             idx = content.lower().find(location.lower())
             start = max(0, idx - 100)
             end = min(len(content), idx + 300)
             snippet = content[start:end]
             return f"⚠️ POSIBLE ALERTA ENCONTRADA para {location}. Fragmento del aviso oficial:\n...{snippet}...\n(Consulta la web de AEMET para detalles oficiales)."
        else:
             return f"No encontré menciones explícitas de alertas para '{location}' en el boletín de hoy de AEMET. (Nota: Esto es una búsqueda básica en el texto del aviso)."

    except Exception as e:
        return f"Error consultando alertas AEMET: {str(e)}"

time_tool = Tool(
    name="get_current_time",
    func=get_current_time,
    description="Útil para saber la fecha, hora actual o qué día es hoy."
)

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Útil para saber el clima, temperatura o pronóstico del tiempo de una ciudad o lugar."
)

aemet_tool = Tool(
    name="check_aemet_alerts",
    func=check_aemet_alerts,
    description="Útil para comprobar si hay alertas meteorológicas oficiales (AEMET) de lluvia, viento, nieve, etc. en una provincia o ciudad española."
)

tools = [time_tool, weather_tool, aemet_tool]

# --- Prompt ---
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

try:
    # Inicializar el LLM
    llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key, temperature=0)

    # Crear el agente
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

except Exception as e:
    st.error(f"Error al inicializar el agente: {e}")
    st.stop()

# --- Gestión del Historial de Chat ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu agente meteorológico. Pregúntame sobre el clima o si hay alertas en tu zona."}
    ]

# Mostrar mensajes del historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Ejecución del Agente ---

if prompt_input := st.chat_input("Escribe tu pregunta aquí..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Respuesta del asistente
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Ejecutar el agente con el callback de Streamlit
            response = agent_executor.invoke(
                {"input": prompt_input},
                {"callbacks": [st_callback]}
            )
            output_text = response["output"]
            
            st.markdown(output_text)
            
            # Guardar respuesta en el historial
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Lo siento, ocurrió un error: {e}"})
