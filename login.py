# import libraries
import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_supabase_auth import login_form, logout_button
from supabase import create_client, Client
from st_supabase_connection import SupabaseConnection

# CSS to hide the sidebar when the user is not in session
hide_sidebar = """
<style>
    section[data-testid="stSidebar"][aria-expanded="true"] {
        display: none;
    }
</style>
"""

def main():
    # Access the session state
    session_state = st.session_state

    # Set up the Streamlit app title and header
    st.title("Login")
    st.header("Auth using Supabase")



    # Load Supabase credentials from environment variables
    load_dotenv()
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_api_key = os.environ.get("SUPABASE_KEY")

    # Check if Supabase credentials are available
    if not (supabase_url and supabase_api_key):
        st.error("Supabase credentials are missing. Please check your environment.")
    else:
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_api_key)

        # Access the session state
        session_state = st.session_state



        @st.cache
        def init_connection():
            return supabase

    # Get user session information using Supabase login form
    session = login_form(
        url=supabase_url,
        apiKey=supabase_api_key,
        providers=["github", "google"],
    )

    # If user is not in session, prompt them to login
    if not session:
        print('USER NOT IN SESSION, LOGIN TO CONTINUE')
        st.markdown(hide_sidebar, unsafe_allow_html=True)
        return
    else:
        # If user is in session, retrieve user details
        print('USER SESSION EXISTS - Redirecting to APPLICATIONPAGE')
        if 'id' not in session_state:
            # Set user details in session state
            session_state.id = session['user']['id']
            session_state.email = session['user']['email']
            # Redirect to the home page
            st.switch_page('pages/aqi_app.py')

        # Display user information in the sidebar
        st.write(f"Welcome  - {session_state.email}")
        # Add a logout button to the sidebar
        logout_button()



def clearSessionState():
    # Clear the previous session state if it exists
    prev_session_state = st.session_state
    if hasattr(st, 'prev_session_state'):
        print('Old Session state Exists -- Clearing')
        prev_session_state.id = None
        prev_session_state.email = None
        print('Old Session state Cleared.')


if __name__ == "__main__":
    # Clear the session state at the beginning
    clearSessionState()
    # Run the main function
    main()
