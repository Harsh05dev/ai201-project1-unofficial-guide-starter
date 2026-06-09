"""Central settings for The Unofficial Guide (NJIT CS professor reviews)."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Embeddings (local, no API key) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector store ---
CHROMA_COLLECTION = "unofficial_guide"
CHROMA_PATH = "./chroma_db"

# --- Retrieval ---
N_RESULTS = 5  # top-k chunks per query

# --- Documents ---
DOCS_PATH = "./documents"
