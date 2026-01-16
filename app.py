import streamlit as st
from ui import render_sidebar, render_main_content

def main():
    st.set_page_config("GA03 Research Intelligence", layout="wide")
    st.title("📚 GA03 – Research Paper Management & Analysis Intelligence System")

    # Initialize State
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Load Modules
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()