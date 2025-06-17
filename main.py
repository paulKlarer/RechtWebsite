import streamlit as st
import google.generativeai as genai
from google import genai as genai_new  # New API for grounding
import openai
import os
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    HttpOptions,
    Tool,
)
BACKGROUND_INFO_FILE_PATH = os.path.join(os.path.dirname(__file__), "background.txt")

# --- Configuration ---
# Load credentials from Streamlit secrets
# No fallbacks - credentials must be in secrets.toml
# APP_USERNAME_SECRET = st.secrets.get("APP_USERNAME") # Removed fallback
# APP_PASSWORD_SECRET = st.secrets.get("APP_PASSWORD") # Removed fallback
# These variables are not strictly needed at the global scope anymore
# as we directly check st.secrets within functions.

# Available Models (Mix of Gemini and OpenAI)
AVAILABLE_MODELS = {
    "Google Gemini": [
        "gemini-2.0-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-03-25"
    ],
    "OpenAI GPT": [
        "o3-mini-2025-01-31",
        "gpt-4.1-2025-04-14",
        "o3-2025-04-16"      
    ]
}
FLAT_AVAILABLE_MODELS = [model for provider_models in AVAILABLE_MODELS.values() for model in provider_models]

# --- API Key Configuration ---
GEMINI_API_CONFIGURED = False
OPENAI_API_CONFIGURED = False

try:
    google_api_key_secret = st.secrets.get("GOOGLE_API_KEY")
    if google_api_key_secret:
        genai.configure(api_key=google_api_key_secret)
        GEMINI_API_CONFIGURED = True
    else:
        GEMINI_API_CONFIGURED = False
except (FileNotFoundError, KeyError): # Handles cases where secrets file might not exist or key is missing
    GEMINI_API_CONFIGURED = False

try:
    openai_api_key_secret = st.secrets.get("OPENAI_API_KEY")
    if openai_api_key_secret:
        OPENAI_API_CONFIGURED = True
    else:
        OPENAI_API_CONFIGURED = False
except (FileNotFoundError, KeyError): # Handles cases where secrets file might not exist or key is missing
    OPENAI_API_CONFIGURED = False


# --- Authentication Functions ---
def check_login(username, password):
    """Validates user credentials against those from secrets."""
    # Ensure secrets are present before attempting to retrieve them.
    # This check is also done in show_login_form, but good for robustness.
    if "APP_USERNAME" not in st.secrets or "APP_PASSWORD" not in st.secrets:
        # This case should ideally be handled by show_login_form preventing the call
        st.warning("Attempted login check, but app credentials are not in secrets.")
        return False
    
    expected_username = st.secrets["APP_USERNAME"] # Direct access, assumes check prior
    expected_password = st.secrets["APP_PASSWORD"] # Direct access, assumes check prior
    
    return username == expected_username and password == expected_password

def show_login_form():
    """Displays the login form."""
    st.sidebar.title("Login")
    
    # Critical check: ensure credentials are in secrets before showing form
    if "APP_USERNAME" not in st.secrets or "APP_PASSWORD" not in st.secrets:
        st.sidebar.error("Critical: App credentials (APP_USERNAME, APP_PASSWORD) are not set in .streamlit/secrets.toml.")
        st.info("Login is disabled. Please contact the administrator to configure application credentials in the Streamlit secrets file.")
        return # Prevent login form if secrets are not set

    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if check_login(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username # Store the entered username
                if "selected_model" not in st.session_state:
                    st.session_state["selected_model"] = FLAT_AVAILABLE_MODELS[0]
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")

def show_logout_button():
    """Displays the logout button."""
    if st.sidebar.button("Logout"):
        del st.session_state["authenticated"]
        del st.session_state["username"]
        st.rerun()

def show_clear_chat_button():
    """Displays the clear chat button."""
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- LLM Interaction Function ---
def get_llm_response(chat_history, model_name):
    """
    Sends a prompt (with history) to the specified LLM and returns the response.
    """
    provider = None
    if model_name in AVAILABLE_MODELS["Google Gemini"]:
        provider = "Google Gemini"
    elif model_name in AVAILABLE_MODELS["OpenAI GPT"]:
        provider = "OpenAI GPT"

    if provider == "Google Gemini":
        if not GEMINI_API_CONFIGURED:
            st.error("Google API Key not configured for Gemini models.")
            return None
        try:
            # Check if this is the grounding-enabled model
            if model_name == "gemini-2.5-flash-preview-05-20":
                # Use new API with Google Search grounding
                client = genai_new.Client(
                    api_key=st.secrets.get("GOOGLE_API_KEY"),
                    http_options=HttpOptions(api_version="v1alpha")  # Try v1alpha instead of v1
                )
                
                # Convert chat history to simple string format for new API
                if chat_history:
                    last_message = chat_history[-1]["content"] if chat_history else ""
                else:
                    last_message = ""
                
                # Try different configuration structure
                response = client.models.generate_content(
                    model=model_name,
                    contents=last_message,
                    config=GenerateContentConfig(
                        tools=[Tool(google_search=GoogleSearch())],
                        response_modalities=["TEXT"]  # Explicitly specify text response
                    ),
                )
            else:
                # Use old API for other Gemini models
                gemini_history = []
                for msg in chat_history:
                    role = "model" if msg["role"] == "assistant" else msg["role"]
                    gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(gemini_history)
            
            return response.text
        except Exception as e:
            st.error(f"Error with Google Gemini API ({model_name}): {e}")
            return None
            
    elif provider == "OpenAI GPT":
        if not OPENAI_API_CONFIGURED:
            st.error("OpenAI API Key not configured for OpenAI models.")
            return None
        try:
            # Ensure OPENAI_API_KEY is fetched from secrets directly here for safety
            # as openai client might not pick it up from global if not explicitly passed
            client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=chat_history
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error with OpenAI API ({model_name}): {e}")
            return None
    else:
        st.error(f"Unknown model provider for: {model_name}")
        return None
ADDITIONAL_LLMS ="""
https://www.s2-ai.com/chat
https://www.phind.com/
https://t3.chat/
https://lmarena.ai/?mode=direct
https://grok.com/?referrer=website

"""
def show_background_info_page():
    st.title("Background Information")
    st.markdown(background_info_content)
    # Add a button to go back to the chat
    if st.sidebar.button("← Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()
    st.sidebar.markdown("---") # Separator
    with st.sidebar.expander("Additional LLMs"):
        st.markdown(ADDITIONAL_LLMS)


# --- Main Application ---
try:
    with open(BACKGROUND_INFO_FILE_PATH, "r", encoding="utf-8") as f:
        background_info_content = f.read()
except FileNotFoundError:
    background_info_content = "Error: background.txt not found!"
except Exception as e:
    background_info_content = f"Error reading background.txt: {e}"

def  main_chat_app():
    """Main application logic after authentication."""
    st.sidebar.success(f"Logged in as: {st.session_state['username']}")
    show_logout_button()
    show_clear_chat_button()

    st.session_state.selected_model = st.sidebar.selectbox(
        "Choose an LLM Model:",
        FLAT_AVAILABLE_MODELS,
        index=FLAT_AVAILABLE_MODELS.index(st.session_state.get("selected_model", FLAT_AVAILABLE_MODELS[1]))
    )
    st.sidebar.markdown("---") # Separator

    # Button to switch to Background Info page
    if st.sidebar.button("Background Info"):
        st.session_state.page = "background_info"
        st.rerun()

    # Button to go back to Chat (only if not already on chat page)
    if st.session_state.page != "chat" and st.sidebar.button("Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()
    st.sidebar.markdown("---") # Separator
    with st.sidebar.expander("Additional LLMs"):
        st.markdown(ADDITIONAL_LLMS)

    st.sidebar.markdown("---") # Separator

    st.title("GOOOOOGLE")
    st.write(f"Using model: `{st.session_state.selected_model}`")

    selected_is_gemini = st.session_state.selected_model in AVAILABLE_MODELS["Google Gemini"]
    selected_is_openai = st.session_state.selected_model in AVAILABLE_MODELS["OpenAI GPT"]

    # Display API key warnings
    if not GEMINI_API_CONFIGURED and selected_is_gemini:
        st.warning(
            "Google API Key is not configured. Selected Gemini model may not work. "
            "Please create/update `.streamlit/secrets.toml` with your `GOOGLE_API_KEY`."
        )
    if not OPENAI_API_CONFIGURED and selected_is_openai:
        st.warning(
            "OpenAI API Key is not configured. Selected OpenAI model may not work. "
            "Please create/update `.streamlit/secrets.toml` with your `OPENAI_API_KEY`."
        )
    
    # General warning if no API keys are configured at all for any provider.
    # This is a bit redundant if specific warnings above are shown, but can be a catch-all.
    if not GEMINI_API_CONFIGURED and not OPENAI_API_CONFIGURED:
         st.error("Neither Google nor OpenAI API keys are configured. The app will likely not function with any model.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"LLM ({st.session_state.selected_model}) is thinking... 🤔"):
                assistant_response = get_llm_response(st.session_state.messages, st.session_state.selected_model)
            
            if assistant_response:
                full_response = assistant_response
                message_placeholder.markdown(full_response)
            else:
                full_response = "Sorry, I couldn't get a response from the LLM."
                message_placeholder.markdown(full_response)
        
        if assistant_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- App Entry Point ---
if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = FLAT_AVAILABLE_MODELS[1]   
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "page" not in st.session_state: # New: Default page is chat
        st.session_state["page"] = "chat"

        st.markdown("---") # Separator
    # Check if essential app credentials are in secrets before showing login
    app_creds_configured = "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets

    if st.session_state["authenticated"]:
        # Only show the page if authenticated
        if st.session_state["page"] == "chat":
            main_chat_app() # Render the chat application
        elif st.session_state["page"] == "background_info":
            show_background_info_page() # Render the background info page
    else:
        # User is not authenticated - always show login or error, and reset page to 'chat'
        st.session_state["page"] = "chat" # Force page back to chat/login view if not authenticated
        if app_creds_configured:
            show_login_form()
            st.info("Please log in using the credentials from your secrets.toml file.")
        else:
            st.sidebar.title("Login")
            st.sidebar.error("Critical: App credentials (APP_USERNAME, APP_PASSWORD) are not set in .streamlit/secrets.toml.")
            st.error("Application login is disabled because credentials are not configured in secrets. Please contact the administrator.")

