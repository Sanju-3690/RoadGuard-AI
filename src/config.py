import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def get_secret(name, default=""):
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return os.getenv(name, default)


gemini_api_key = get_secret("GEMINI_API_KEY", "")
gemini_model = get_secret("GEMINI_MODEL", "gemini-3.8-flash")