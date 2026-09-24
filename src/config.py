"""
Module de configuration centralisée de l'application IAR-API.

Ce module extrait et valide les variables d'environnement en utilisant la bibliothèque
`python-dotenv`. Il garantit qu'aucun secret (clés API LLM, mots de passe) n'est codé
en dur dans le code source et fournit des valeurs par défaut sécurisées.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Recherche du fichier .env à la racine du projet
_BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BASE_DIR / ".env"

# Chargement des variables depuis le fichier .env si présent
if _ENV_FILE.exists():
    load_dotenv(dotenv_path=_ENV_FILE)
else:
    # Chargement standard des variables d'environnement système
    load_dotenv()


class Settings:
    """
    Paramètres globaux de configuration de l'API RAMEAU.
    """

    # --- Configuration Qdrant ---
    QDRANT_HOST: str = os.getenv("IAR_QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("IAR_QDRANT_PORT", "6333"))

    # --- Configuration LLM ABES ---
    ABES_LLM_URL: str = os.getenv("IAR_LLM_URL", "http://iar-llm:11434/v1").strip()
    ABES_LLM_KEY: str = os.getenv("IAR_LLM_KEY", "").strip()
    ABES_LLM_MODEL: str = os.getenv("IAR_LLM_MODEL", "llama-3.1-8b").strip()

    # --- Configuration Serveur FastAPI ---
    API_HOST: str = os.getenv("IAR_API_HTTP_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("IAR_API_HTTP_PORT", "8071"))

    # --- Sécurité & CORS ---
    _raw_cors: str = os.getenv("IAR_CORS_ORIGINS", "*")
    CORS_ORIGINS: List[str] = [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]

    # --- Accélération matérielle (GPU) ---
    ENABLE_GPU: bool = os.getenv("ENABLE_GPU", "true").lower() in ("true", "1", "yes")


settings = Settings()
