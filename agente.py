import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- Configuración de la página ---
st.set_page_config(page_title="Agente Inteligente con LangChain", page_icon="🤖", layout="wide")

st.title("🤖 FASTANS")
st.markdown("""
Este es un agente que utiliza **Google Gemini** para razonar y **DuckDuckGo** para buscar información reciente en internet.
""")

# --- Sidebar para configuración ---
with st.sidebar:
    st.header("Configuración")
    
    # Input para la API Key de Google
    google_api_key = st.text_input("Google API Key", type="password", key="google_api_key")
    st.markdown("[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)")
    
    selected_model = st.selectbox("Modelo", ["gemini-2.5-flash", "gemini-1.0-pro"], index=0)
    
    st.divider()
    st.markdown("### Acerca de")
    st.markdown("Creado con LangChain y Streamlit.")

# --- Inicialización del Agente ---

if not google_api_key:
    st.info("👋 Por favor, ingresa tu **Google API Key** en la barra lateral para comenzar.")
    st.stop()

if google_api_key:
    selected_model = "gemini-2.5-flash"  # Restore valid model

# Definir herramientas
tools = []

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    tools.append(search)
except ImportError as e:
    st.warning(f"⚠️ No se pudo cargar la herramienta de búsqueda: {e}. El agente funcionará sin búsqueda.")
except Exception as e:
    st.warning(f"⚠️ Error al inicializar la búsqueda: {e}. El agente funcionará sin búsqueda.")


# Definir el prompt
# Usamos un prompt estándar para ReAct
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
        {"role": "assistant", "content": "¡Hola! Soy un agente inteligente. Puedo buscar en internet y responder tus preguntas. ¿En qué te ayudo hoy?"}
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
