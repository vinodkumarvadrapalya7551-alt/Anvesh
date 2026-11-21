"""
Configuration module for SkillBridge EdTech Agentic RAG System.
Handles paths, constants, reference dates, and API key management with key rotation.
"""

import os
import time
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, *args, **kwargs):
        """Fallback .env parser if python-dotenv is not yet installed in active environment."""
        p = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parent.parent / ".env"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag_config")

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
VECTOR_STORE_DIR = ROOT_DIR / "vector_store_cache"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(ROOT_DIR / ".env")

# --------------------------------------------------------------------------
# CRITICAL COMPETITION CONSTANTS
# --------------------------------------------------------------------------
# The reference date for all calculations is January 15, 2025.
# Any date-relative computation (weeks since enrollment, progress, etc.)
# MUST use this date.
REFERENCE_DATE = date(2025, 1, 15)
REFERENCE_DATE_STR = "2025-01-15"
REFERENCE_DATETIME = datetime(2025, 1, 15, 0, 0, 0)

# Models
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")


# --------------------------------------------------------------------------
# GROQ KEY MANAGER WITH AUTOMATIC ROTATION
# --------------------------------------------------------------------------
class GroqKeyManager:
    """
    Manages multiple Groq API keys and rotates them automatically
    when rate limits (HTTP 429 / RateLimitError) occur.
    """
    def __init__(self):
        self.keys: List[str] = self._discover_keys()
        self.current_index = 0
        self.key_cooldowns: dict = {}  # key -> cooldown end timestamp
        self._groq_client = None

        if not self.keys:
            logger.warning(
                "No GROQ API keys detected in environment. "
                "Please set GROQ_API_KEY, GROQ_API_KEY_1, etc. in .env"
            )
        else:
            logger.info(f"Loaded {len(self.keys)} Groq API key(s) for rotation.")

    def _discover_keys(self) -> List[str]:
        """Discovers all GROQ_API_KEY variables in environment."""
        keys = []
        # Check standard key
        single_key = os.getenv("GROQ_API_KEY")
        if single_key and single_key.strip():
            keys.append(single_key.strip())

        # Check numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, ...
        idx = 1
        while True:
            k = os.getenv(f"GROQ_API_KEY_{idx}")
            if not k:
                # also check non-prefixed or lower
                break
            if k.strip() and k.strip() not in keys:
                keys.append(k.strip())
            idx += 1

        # Also search for any env var matching GROQ_API_KEY*
        for env_name, val in os.environ.items():
            if env_name.startswith("GROQ_API_KEY") and val.strip() and val.strip() not in keys:
                keys.append(val.strip())

        return keys

    @property
    def has_keys(self) -> bool:
        return len(self.keys) > 0

    def get_current_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.current_index]

    def get_client(self):
        """Returns an initialized Groq client using the active key."""
        from groq import Groq
        current_key = self.get_current_key()
        if not current_key:
            raise ValueError("No Groq API key available. Please add GROQ_API_KEY to your environment.")
        return Groq(api_key=current_key)

    def rotate_key(self, cooldown_seconds: int = 60):
        """
        Rotates to the next available API key and applies cooldown
        to the exhausted key.
        """
        if len(self.keys) <= 1:
            logger.warning("Only 1 Groq API key available. Waiting 5s before retrying...")
            time.sleep(5)
            return self.get_client()

        old_idx = self.current_index
        old_key = self.keys[old_idx]
        self.key_cooldowns[old_key] = time.time() + cooldown_seconds

        # Advance to next key
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        logger.info(
            f"Rotated Groq key: Key #{old_idx + 1} (ended with ...{old_key[-4:]}) "
            f"-> Key #{self.current_index + 1} (ended with ...{new_key[-4:]})"
        )
        return self.get_client()


# --------------------------------------------------------------------------
# GOOGLE GEMINI CLIENT FACTORY
# --------------------------------------------------------------------------
def get_google_api_key() -> Optional[str]:
    """Retrieve Google Gemini API Key from environment."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def get_gemini_client():
    """Initializes and returns google-genai Client."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "The 'google-genai' SDK is not installed in the active Python environment. "
            "Please install it with: pip install google-genai"
        )
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set in environment or .env."
        )
    return genai.Client(api_key=api_key)


# Global configuration instance
class AppConfig:
    REFERENCE_DATE = REFERENCE_DATE
    REFERENCE_DATE_STR = REFERENCE_DATE_STR
    REFERENCE_DATETIME = REFERENCE_DATETIME
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    POLICIES_DIR = POLICIES_DIR
    VECTOR_STORE_DIR = VECTOR_STORE_DIR
    LOGS_DIR = LOGS_DIR
    DEFAULT_GROQ_MODEL = DEFAULT_GROQ_MODEL
    DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL

    def __init__(self):
        self.groq_keys = GroqKeyManager()


config = AppConfig()
