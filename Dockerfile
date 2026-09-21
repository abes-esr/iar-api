# ==============================================================================
# Dockerfile - API RAMEAU (FastAPI)
# Image de conteneur optimisée et sécurisée pour l'indexation par IA
# ==============================================================================

FROM python:3.10-slim

# Empêcher Python d'écrire des fichiers .pyc et forcer l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    PORT=8071

# Installation des dépendances système (compilateurs pour omikuji, curl pour le healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-privilégié (sécurité renforcée)
RUN useradd -m -u 10001 appuser

WORKDIR /app

# Mise en cache optimale des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pré-téléchargement des ressources linguistiques NLTK légères nécessaires
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Copie du code applicatif
COPY src/ ./src/

# Création des répertoires de traces avec permissions pour l'utilisateur applicatif
RUN mkdir -p /app/responses /app/data \
    && chown -R appuser:appuser /app

# Bascule vers l'utilisateur non-root
USER appuser

# Exposition du port du service
EXPOSE 8071

# Vérification périodique de l'état de santé du conteneur
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8071/health || exit 1

# Démarrage du serveur Uvicorn
CMD ["python", "-m", "uvicorn", "src.rameau:app", "--host", "0.0.0.0", "--port", "8071"]
