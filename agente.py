import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
import datetime
import requests

# --- Configuración de la página ---
st.set_page_config(page_title="Agente de Tiempo y Clima", page_icon="☀️", layout="wide")

st.title("☀️ Agente Meteorológico ☁️")
st.markdown("""
Este agente te dice la hora exacta y el clima en cualquier parte del mundo usando **Open-Meteo**.
""")

# --- Sidebar para configuración ---
with st.sidebar:
    st.header("Configuración")
    
    # Input para la API Key de Google
    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key")
    st.markdown("[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)")
    
    # Modelos actualizados
    model_options = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
    selected_model = st.selectbox("Modelo", model_options, index=0)
    
    st.divider()
    st.markdown("### Acerca de")
    st.markdown("Creado con LangChain, Streamlit y Open-Meteo API.")

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
    Obtiene el clima actual y pronóstico para una ubicación dada.
    Usa Open-Meteo API (gratuita).
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
        
        # Simple mapping for weather codes (WMO)
        # https://open-meteo.com/en/docs
        
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

tools = [time_tool, weather_tool]

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
        {"role": "assistant", "content": "¡Hola! Soy tu agente meteorológico. Pregúntame la hora o el clima de cualquier ciudad."}
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
