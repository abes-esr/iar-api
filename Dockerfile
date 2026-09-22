# ==============================================================================
# Étape de Base (base)
# ==============================================================================
FROM python:3.10-slim AS base

# Empêcher la création de fichiers .pyc et activer la sortie immédiate des logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/src \
    PORT=8071

WORKDIR /app

# Installation des dépendances système :
# - build-essential : requis pour la compilation d'omikuji (C++)
# - curl : requis pour le healthcheck Docker
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pré-téléchargement des ressources linguistiques NLTK requises par embed_lib
RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Création du groupe docker et de l'utilisateur non-root (UID 1000 pour la compatibilité avec les volumes hôte)
RUN (groupadd -g 999 docker 2>/dev/null || groupadd docker) && \
    useradd -u 1000 -m -s /bin/bash appuser && \
    usermod -aG docker appuser && \
    mkdir -p /app/data /app/volumes /app/responses && \
    chown -R appuser:appuser /app

# ==============================================================================
# Étape de Production (api-image)
# ==============================================================================
FROM base AS api-image

# Copie des sources applicatives avec les permissions directes pour appuser
COPY --chown=appuser:appuser src/ /app/src/

# Liens symboliques pour faciliter l'exécution directe des modules à la racine
RUN ln -s /app/src/rameau.py /app/rameau.py && \
    ln -s /app/src/config.py /app/config.py && \
    ln -s /app/src/embed_lib.py /app/embed_lib.py && \
    ln -s /app/src/omk.py /app/omk.py

USER appuser

EXPOSE 8071

# Vérification périodique de l'état de santé du service (start-period=45s pour le chargement des modèles HuggingFace)
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8071/health || exit 1

# Commande par défaut : lancement du serveur FastAPI avec Uvicorn
CMD ["uvicorn", "rameau:app", "--host", "0.0.0.0", "--port", "8071"]

