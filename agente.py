import streamlit as st
import datetime
import requests
import xml.etree.ElementTree as ET
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

# --- 1. Configuración de Página y Estilos CSS Premium ---
st.set_page_config(
    page_title="Meteorolog.IA",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS avanzados: Glassmorphism, Neumorphism, Animaciones
st.markdown("""
<style>
    /* Importar fuente futurista */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');

    /* Variables de tema */
    :root {
        --primary-color: #00d2ff;
        --secondary-color: #3a7bd5;
        --bg-color: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --text-color: #e2e8f0;
        --accent: #f59e0b;
    }

    /* Reset y base */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text-color);
    }
    
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(58, 123, 213, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(245, 158, 11, 0.1) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Títulos con gradiente */
    h1, h2, h3 {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Tarjetas Glassmorphism */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* Sidebar personalizado */
    section[data-testid="stSidebar"] {
        background-color: #0a0e17;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Chat Messages */
    .stChatMessage {
        background: transparent;
        border: none;
    }
    [data-testid="stChatMessageContent"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input de chat estilizado */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .stChatInputContainer input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        border-radius: 24px !important;
    }
    
    /* Ocultar elementos default de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Animación de carga sutil para el spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Funciones Lógicas (Backend) ---

def get_current_time(query: str = "") -> str:
    """Devuelve la fecha y hora actual exacta. Úsala cuando pregunten 'qué hora es' o 'qué día es hoy'."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_weather(location: str) -> str:
    """Obtiene el clima actual y pronóstico para una ciudad usando Open-Meteo API.
    Devuelve un string detallado con temperatura, viento y máximas/mínimas."""
    try:
        # 1. Geocoding
        # Limpiamos la location para evitar caracteres raros
        location = location.strip()
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=es&format=json"
        
        try:
            geo_res = requests.get(geo_url, timeout=5).json()
        except Exception:
            return f"Error de conexión al buscar la ubicación '{location}'."

        if not geo_res.get("results"):
            return f"No encontré la ubicación '{location}'. Por favor verifica el nombre."
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        name = geo_res["results"][0]["name"]
        country = geo_res["results"][0].get("country", "")
        
        # 2. Weather Data
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset"
            "&timezone=auto"
        )
        
        try:
            weather_res = requests.get(weather_url, timeout=5).json()
        except Exception:
            return "Error conectando con el servicio de clima."
        
        current = weather_res.get("current", {})
        daily = weather_res.get("daily", {})
        current_units = weather_res.get("current_units", {})
        
        # Datos Actuales
        temp = current.get("temperature_2m", "N/A")
        feels_like = current.get("apparent_temperature", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        
        # Datos Diarios (Hoy)
        if daily and 'temperature_2m_max' in daily:
            max_temp = daily['temperature_2m_max'][0]
            min_temp = daily['temperature_2m_min'][0]
            sunrise = daily.get('sunrise', [''])[0][-5:] # Solo la hora
            sunset = daily.get('sunset', [''])[0][-5:]
        else:
            max_temp = min_temp = sunrise = sunset = "N/A"

        # Formateo de respuesta estructurada para el LLM
        report = (
            f"📍 **Informe Climático para {name}, {country}**\n"
            f"🌡️ **Actual:** {temp}{current_units.get('temperature_2m','°C')} (Sensación: {feels_like}°)\n"
            f"💧 **Humedad:** {humidity}%\n"
            f"💨 **Viento:** {wind} km/h\n"
            f"📅 **Pronóstico Hoy:** Máx {max_temp}° / Mín {min_temp}°\n"
            f"☀️ **Sol:** Sale {sunrise} / Pone {sunset}\n"
        )
        return report

    except Exception as e:
        return f"Ocurrió un error inesperado al obtener el clima: {str(e)}"

def search_func(query: str) -> str:
    """Busca en internet usando DuckDuckGo como fallback."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # Intentamos backend 'lite' que suele ser más rápido/estable
            results = list(ddgs.text(query, max_results=3, backend="lite"))
            if results:
                summary = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                return f"Resultados de búsqueda:\n{summary}"
        return "No encontré información relevante en la búsqueda rápida."
    except Exception as e:
        return f"Error en búsqueda web: {str(e)}"

def check_aemet_alerts(location: str) -> str:
    """Verifica alertas oficiales de AEMET. Requiere API Key en configuración."""
    
    # Recuperamos la key del estado de sesión si no se pasa explícitamente (el agente no pasa keys)
    api_key = st.session_state.get("aemet_api_key", "")
    
    if not api_key:
        return f"⚠️ No tengo configurarada la API Key de AEMET. Buscando noticias recientes sobre alertas en {location}...\n" + search_func(f"Alertas meteorológicas AEMET {location} última hora")

    try:
        requests.packages.urllib3.disable_warnings() 
        # Endpoint de avisos de hoy
        url = "https://opendata.aemet.es/opendata/api/avisos_de_fenomenos_meteorologicos_adversos/archivo/hoy"
        
        # 1. Obtener URL de datos
        res = requests.get(url, params={"api_key": api_key}, verify=False, timeout=10)
        
        if res.status_code == 200:
            json_res = res.json()
            if json_res.get("estado") == 200:
                data_url = json_res.get("datos")
                # 2. Descargar datos reales
                data_res = requests.get(data_url, verify=False, timeout=15)
                data_res.encoding = 'iso-8859-15' # AEMET suele usar esta codificación o utf-8
                content = data_res.text
                
                # Búsqueda simple en el texto raw (podría mejorarse parseando JSON si AEMET lo devuelve estructurado ahí, pero a veces es XML/TXT)
                # El endpoint 'archivo/hoy' suele devolver un JSON grande con lista de avisos.
                
                # Intentamos parsear si es json
                try:
                    alerts_data = data_res.json()
                    # Si es una lista de avisos, filtramos
                    relevant_alerts = []
                    # Esta lógica depende mucho de la estructura exacta de AEMET que es compleja
                    # Haremos un escaneo de texto simple para robustez inmediata
                    if location.lower() in str(alerts_data).lower():
                         return f"🚨 **ALERTA DETECTADA EN AEMET** para la región de {location}. Se recomienda precaución. Consulta la web oficial para detalles específicos."
                    else:
                        return f"✅ No se detectan avisos específicos mencinando '{location}' en el boletín de hoy de AEMET."
                except:
                    # Fallback texto
                    if location.lower() in content.lower():
                        return f"🚨 Mención encontrada en boletín AEMET para {location}. Posible alerta."
                    else:
                         return f"Info AEMET consultada. No parece haber alertas graves mencionando explícitamente {location}."
            else:
                return f"Error AEMET: {json_res.get('descripcion')}"
        elif res.status_code == 401:
            return "Error: AEMET API Key inválida."
        else:
            return f"Error de conexión con AEMET (Status {res.status_code})."

    except Exception as e:
        return search_func(f"Alertas clima {location}")

# --- 3. Definición de Herramientas y Agente ---

tools = [
    Tool(
        name="get_current_time",
        func=get_current_time,
        description="Usa esto para obtener la fecha y hora actual. Input: string vacío."
    ),
    Tool(
        name="get_weather",
        func=get_weather,
        description="Usa esto para obtener el clima actual y pronóstico. Input: nombre de la ciudad (ej: 'Madrid')."
    ),
    Tool(
        name="check_aemet_alerts",
        func=check_aemet_alerts,
        description="Usa esto SOLO para verificar alertas de seguridad oficiales en España (AEMET). Input: nombre de la ciudad/región."
    )
]

# Prompt mejorado con "Persona" y "Memoria"
template = """Eres 'Meteorolog.IA', una asistente experta en meteorología y clima, profesional pero amable y con un toque futurista.

Tu objetivo es dar información climática precisa y útil.
1. SIEMPRE usa las herramientas si te preguntan por datos reales (clima, hora, alertas). No inventes.
2. Si el usuario saluda, responde amablemente y ofrece tu ayuda.
3. Si detectas una alerta o clima peligroso, da recomendaciones de seguridad.
4. Usa formato Markdown atractivo (negritas, emojis) en tu respuesta final.
5. Piensa paso a paso.

Historial de conversación:
{chat_history}

Pregunta del usuario: {input}

Herramientas disponibles:
{tools}

Usa el siguiente formato:

Question: la pregunta del usuario
Thought: piensa qué hacer (verificar historial, usar herramienta, o responder directo)
Action: la herramienta a usar (una de [{tool_names}])
Action Input: el input para la herramienta
Observation: el resultado de la herramienta
... (repite Thought/Action/Observation si es necesario)
Thought: ya tengo la respuesta final
Final Answer: la respuesta final al usuario

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# --- 4. Interfaz de Usuario (Sidebar & Main) ---

with st.sidebar:
    st.markdown("<div style='text-align: center;'><h1>🌪️ Meteorolog.IA</h1></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🔐 Credenciales API", expanded=True):
        st.caption("Configura tus llaves para activar el cerebro del agente.")
        google_api_key = st.text_input("Google Gemini Key", type="password", key="google_key_input").strip()
        st.session_state.aemet_api_key = st.text_input("AEMET API Key (Opcional)", type="password", key="aemet_key_input").strip()
        
        col1, col2 = st.columns(2)
        with col1:
             st.markdown("[Google Key ↗](https://aistudio.google.com/app/apikey)")
        with col2:
             st.markdown("[AEMET Key ↗](https://opendata.aemet.es/centrodescargas/inicio)")

    with st.expander("🤖 Configuración del Modelo", expanded=False):
        # Restauramos modelos válidos conocidos.
        model_options = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        selected_model = st.selectbox("Versión del Modelo", model_options, index=0)
        temperature = st.slider("Creatividad", 0.0, 1.0, 0.0)

    st.markdown("---")
    st.info("💡 **Pro Tip:** Prueba preguntar '¿Hay alertas en Valencia hoy?' o '¿Qué tiempo hará mañana en Barcelona?'")
    
    # Botón de reset memoria
    if st.button("🗑️ Borrar Memoria"):
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
        st.session_state.messages = []
        st.rerun()

# --- 5. Lógica Principal del Chat ---

# Inicializar historial visual
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy **Meteorolog.IA**. 🌩️\nEstoy conectada a satélites y estaciones en tiempo real.\n¿En qué ciudad te encuentras hoy?"}
    ]

# Inicializar Memoria de LangChain
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Renderizar mensajes anteriores
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🌪️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input del usuario
if user_input := st.chat_input("Escribe tu consulta climática..."):
    
    if not google_api_key:
        st.warning("⚠️ Error: Por favor ingresa tu **Google API Key** en el menú lateral para continuar.")
        st.stop()
        
    # 1. Mostrar mensaje usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Procesar con Agente
    with st.chat_message("assistant", avatar="🌪️"):
        status_container = st.status("🛰️ Procesando datos satelitales...", expanded=True)
        try:
            # Callback para ver el pensamiento en el expander
            st_cb = StreamlitCallbackHandler(status_container)
            
            # Inicializar LLM y Agente (Re-creado para actualizarse con config)
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                google_api_key=google_api_key,
                temperature=temperature
            )
            
            # Crear Agente con Memoria
            # NOTA: create_react_agent estándar no inyecta memoria automáticamente en agent_scratchpad
            # Usamos AgentExecutor para manejar la memoria
            agent = create_react_agent(llm, tools, prompt)
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                memory=st.session_state.memory, # AQUÍ está la clave de la memoria 🧠
                handle_parsing_errors=True
            )
            
            response = agent_executor.invoke(
                {"input": user_input},
                {"callbacks": [st_cb]}
            )
            
            output_text = response["output"]
            
            status_container.update(label="✅ Análisis Global Completado", state="complete", expanded=False)
            st.markdown(output_text)
            
            # Guardar en historial visual
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            
        except Exception as e:
            status_container.update(label="❌ Error del Sistema", state="error")
            err_str = str(e)
            if "404" in err_str:
                error_msg = f"⚠️ **Error: Modelo no encontrado.**\nEl modelo '{selected_model}' puede no estar disponible o tu API Key no tiene permiso para usarlo.\nPrueba seleccionar 'gemini-1.5-flash'."
            elif "401" in err_str or "API key not valid" in err_str:
                error_msg = "⚠️ **Error: API Key Inválida.**\nVerifica que has copiado correctamente tu Google API Key en la barra lateral."
            elif "429" in err_str:
                error_msg = "⚠️ **Error: Límite de cuota excedido.**\nHas superado el número de peticiones permitidas por Google."
            else:
                error_msg = f"Lo siento, ocurrió un error inesperado:\n`{err_str}`"
            
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
