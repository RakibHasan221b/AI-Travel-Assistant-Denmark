"""Shared helpers for the Streamlit app (Phase 12).

DB connections are opened fresh per call, not cached — matches every other
script in this project, and avoids sharing one psycopg connection across
Streamlit's concurrent user sessions. The embedding model is cached
(st.cache_resource) since reloading it per interaction would be slow.
"""

import os

import psycopg
import streamlit as st
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _config(key: str, default: str | None = None) -> str | None:
    """st.secrets on Streamlit Community Cloud, falls back to env vars locally
    (st.secrets raises StreamlitSecretNotFoundError when no secrets.toml
    exists at all, which is the normal case for local development)."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except StreamlitSecretNotFoundError:
        pass
    return os.environ.get(key, default)


def get_connection():
    conn = psycopg.connect(_config("DATABASE_URL"), connect_timeout=15)
    register_vector(conn)
    return conn


@st.cache_resource(show_spinner=False)
def get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_api_url() -> str:
    return _config("TRIP_PLANNER_API_URL", "http://localhost:8000")


def query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d.name for d in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
