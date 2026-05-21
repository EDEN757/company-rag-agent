import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.environ.get("LLM_MODEL",   "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

DB_HOST     = os.environ.get("PGHOST",     "nv-service-b01d63337fab32ac94f65eb2dc8a62ba")
DB_PORT     = int(os.environ.get("PGPORT", "5432"))
DB_USER     = os.environ.get("PGUSER",     "nuvolos")
DB_PASSWORD = os.environ.get("PGPASSWORD", "nuvolos")
DB_NAME     = os.environ.get("PGDATABASE", "nuvolos")

RERANKER_MODEL    = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RRF           = os.environ.get("USE_RRF",    "true").lower() == "true"
RRF_K             = 60
HYDE_ENABLED      = os.environ.get("HYDE_ENABLED", "true").lower() == "true"
TOP_K_PER_BRANCH  = 8
VEC_WEIGHT        = 0.7
KW_WEIGHT         = 0.3
SCORE_THRESHOLD   = 0.35
SCORE_SCALE       = 4.0
BM25_K1           = 1.5
BM25_B            = 0.75
MAX_AGENT_TURNS   = 6
MAX_HISTORY_TURNS = 10
MAX_NEW_TOKENS    = 2048
TEMPERATURE       = 0.1
MAX_EMBED_CHARS   = 6000
TABLE_DOCS        = "rag_documents"
TABLE_CHUNKS      = "rag_chunks"

BASH_DENY = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\b>\s*/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
]
