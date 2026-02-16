import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
import datetime
import requests
import xml.etree.ElementTree as ET

# --- Configuración de la página y CSS ---
st.set_page_config(page_title="Agente Meteorológico", page_icon="🌤️", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🌤️ AI Weather Assistant")
st.caption("🚀 Powered by Gemini, Open-Meteo & AEMET")

# --- Sidebar para configuración ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    st.header("⚙️ Configuración")
    
    with st.expander("🔑 API Keys", expanded=True):
        google_api_key = st.text_input("Google API Key", type="password", help="Necesaria para el cerebro del agente (Gemini).")
        aemet_api_key = st.text_input("AEMET API Key", type="password", help="Opcional. Para alertas oficiales precisas en España.")
        st.markdown("[Obtener Google Key](https://aistudio.google.com/app/apikey)")
        st.markdown("[Obtener AEMET Key](https://opendata.aemet.es/centrodescargas/inicio)")

    with st.expander("🧠 Modelo IA", expanded=False):
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
        selected_model = st.selectbox("Versión", model_options, index=0)
    
    st.divider()
    st.info("💡 **Tip:** Pregunta por 'Alertas en [Ciudad]' para ver avisos oficiales.")

if not google_api_key:
    st.warning("⚠️ Por favor, introduce tu **Google API Key** en la barra lateral para activar el agente.")
    st.stop()

# --- HERRAMIENTAS (Lógica Preservada) ---

def get_current_time(query: str = "") -> str:
    """Devuelve la fecha y hora actual exacta."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    """Obtiene el clima actual y pronóstico usando Open-Meteo."""
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

# Re-integrar función de búsqueda para el fallback
def search_func(query: str) -> str:
    """Busca en internet con reintentos."""
    try:
        from duckduckgo_search import DDGS
        # print(f"DEBUG: Ejecutando búsqueda DDGS: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4, backend="html"))
                if results: return str(results)
        except: pass
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4, backend="lite"))
                if results: return str(results)
        except: pass
        return "No pude confirmar alertas por internet. Por precaución, revisa www.aemet.es."
    except: return "Error en librería de búsqueda."

def check_aemet_alerts(location: str) -> str:
    """Consulta alertas meteorológicas vigentes en AEMET con fallback a web."""
    if not aemet_api_key:
        return f"INFO: No tienes AEMET API Key. Buscando información pública en la web para {location}..."
        #return search_func(f"Alertas AEMET {location} hoy") # Opción directa si queremos
        
    try:
        requests.packages.urllib3.disable_warnings()
        url = "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/archivo/hoy"
        params = {"api_key": aemet_api_key}
        
        # 1. Intentar API
        try:
            res = requests.get(url, params=params, verify=False, timeout=10)
            if res.status_code == 200:
                json_res = res.json()
                if json_res.get("estado") == 200:
                    data_url = json_res.get("datos")
                    data_res = requests.get(data_url, verify=False, timeout=20)
                    data_res.encoding = data_res.apparent_encoding
                    content = data_res.text
                    
                    # Normalización y búsqueda
                    if location.lower() in content.lower():
                        idx = content.lower().find(location.lower())
                        start = max(0, idx - 100)
                        end = min(len(content), idx + 300)
                        return f"⚠️ ALERTA ENCONTRADA (API) para {location}.\nFragmento:\n...{content[start:end]}..."
                    else:
                        # No encontrado en boletín -> Fallback
                        pass 
        except Exception:
            pass

        # 2. Fallback
        return search_func(f"AEMET avisos {location} hoy última hora")

    except Exception as e:
        return f"Error consultando alertas: {str(e)}"

# --- Configuración del Agente ---

time_tool = Tool(
    name="get_current_time",
    func=get_current_time,
    description="Útil para saber la fecha, hora actual o qué día es hoy."
)

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Útil para saber el clima, temperatura o pronóstico del tiempo de una ciudad."
)

aemet_tool = Tool(
    name="check_aemet_alerts",
    func=check_aemet_alerts,
    description="Útil para comprobar si hay alertas meteorológicas oficiales (AEMET). Si la ciudad no tiene alertas, devuelve info normal."
)

tools = [time_tool, weather_tool, aemet_tool]

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

# --- Inicialización ---
if "agent_executor" not in st.session_state:
    try:
        llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key, temperature=0)
        agent = create_react_agent(llm, tools, prompt)
        st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    except Exception as e:
        st.error("Error iniciando el motor IA.")

# --- Interfaz de Chat ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente climático inteligente. 🌩️\nPregúntame por el tiempo, la hora o alertas de AEMET."}
    ]

# Renderizar historial
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input de usuario
if prompt_input := st.chat_input("¿Qué tiempo hace en Madrid?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="🤖"):
        # Contenedor colapsable para el "pensamiento" del agente
        with st.status("🧠 Analizando servicios meteorológicos...", expanded=False) as status:
            st_cb = StreamlitCallbackHandler(status)
            try:
                # Si cambiaron la key/modelo, hay que reiniciar el executor? 
                # Simplificación: Lo recreamos si no existe o usamos el de sesión.
                # Para asegurar dinamismo con sidebar, mejor recrear 'llm' en cada run o confiar en el rerun de streamlit.
                # Al ser script, se recrea 'llm' arriba. Actualizamos el executor.
                llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=google_api_key, temperature=0)
                agent = create_react_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
                
                response = agent_executor.invoke(
                    {"input": prompt_input},
                    {"callbacks": [st_cb]}
                )
                status.update(label="✅ Análisis completado", state="complete", expanded=False)
                output_text = response["output"]
            except Exception as e:
                status.update(label="❌ Error en el proceso", state="error")
                output_text = f"Lo siento, tuve un problema técnico: {str(e)}"

        # Mostrar respuesta final fuera del expander
        st.markdown(output_text)
        st.session_state.messages.append({"role": "assistant", "content": output_text})
