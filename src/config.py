import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def get_secret(name, default=""):
    """Read a value from Streamlit Cloud secrets first, then .env."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

    return os.getenv(name, default)


class Settings:
    def __init__(self):
        self.root = ROOT

        # Gemini
        self.gemini_api_key = get_secret("GEMINI_API_KEY", "")
        self.gemini_model = get_secret(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )
        self.confidence = 0.25

        # Trained model
        self.model_path = ROOT / "models" / "roadguard.pt"

        # Knowledge base
        self.knowledge_path = (
            ROOT / "data" / "knowledge" / "maintenance_knowledge.jsonl"
        )

        # Database
        self.db_path = ROOT / "data" / "roadguard.db"

        # Keep the old name too, in case another file uses it
        self.database_path = self.db_path


settings = Settings()