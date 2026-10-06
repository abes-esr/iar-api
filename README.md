# 🚀 iar-api : Service Web d'Indexation Sujet RAMEAU

[![Docker Pulls](https://img.shields.io/docker/pulls/abesesr/iar.svg)](https://hub.docker.com/r/abesesr/iar/)
[![Buildx Publish](https://github.com/abes-esr/iar-api/actions/workflows/buildx-pubtodockerhub.yml/badge.svg)](https://github.com/abes-esr/iar-api/actions/workflows/buildx-pubtodockerhub.yml)

Ce dépôt contient le code source de l'API du projet **IAR** (Indexation Automatique RAMEAU), développé par l'**ABES** (Agence Bibliographique de l'Enseignement Supérieur).

Ce composant backend fournit un service web REST exposé via **FastAPI**, permettant de suggérer des vedettes-matières **RAMEAU** Unimarc (zone `606`) à destination des catalogueurs du Sudoc (notamment via WinIBW) et d'interfaces web, enrichies par de l'intelligence artificielle (recherche vectorielle sémantique, réordonnancement par cross-encoder et filtrage par LLM) à partir des métadonnées d'un document (titre, résumé, PPN).

---

## 📌 Sommaire

- [Architecture & Chaîne de traitement](#-architecture--chaîne-de-traitement)
- [Modèles d'Embedding & Stratégies d'Agrégation](#-modèles-dembedding--stratégies-dagrégation)
- [Spécification de l'API](#-spécification-de-lapi)
- [Configuration (.env)](#-configuration-env)
- [Installation & Démarrage](#-installation--démarrage)
- [Supervision & Logs](#-supervision--logs)
- [Sauvegarde & Restauration](#-sauvegarde--restauration)
- [Structure du Projet & Modules](#-structure-du-projet--modules)
- [Tests & Évaluation](#-tests--évaluation)

---

## 🏗 Architecture & Chaîne de traitement

Le serveur reçoit les requêtes de catalogage (ex. WinIBW via VBScript ou requêtes HTTP). La notice est analysée selon la chaîne suivante :

```
┌─────────────────────────────────────────────────────────────┐
│                       Notice Unimarc                        │
│                  (Titre, Résumé, PPN, RCR)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Requête GET /subject_indexation/
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Router                   │
│           Validation RCR / Détection automatique langue     │
└───────┬──────────────────────┬──────────────────────┬───────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Victor 1   │       │   Victor 2   │       │   Victor 3   │
│ (all-MiniLM) │       │ (distiluse)  │       │  (e5-large)  │
└───────┬──────┘       └───────┬──────┘       └───────┬──────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Recherche Vectorielle (Qdrant)                 │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Module de Post-Traitements & Agrégation          │
│   • Intersections & Reranking Cross-Encoder                 │
│   • Filtrage/Sélection LLM (Llama 3.1 via vLLM / Ollama)    │
│   • Structuration Hiérarchique RAMEAU (Arborescence/Roots)  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Format de Sortie (Unimarc 606)                  │
│       Format JSON, HTML (flat/tree) ou Text (flat/tree)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Modèles d'Embedding & Stratégies d'Agrégation

### Modèles IA (_Victor_)

- **`victor1_concept` / `victor1_chain`** : Basé sur `sentence-transformers/all-MiniLM-L6-v2` (rapide, optimisé concepts).
- **`victor2`** : Basé sur `distiluse-base-multilingual-cased-v2` (multilingue compact).
- **`victor3_chain` / `victor3_chain_en`** : Basé sur `intfloat/multilingual-e5-large` (haute précision, multilingue).

### Stratégies d'Agrégation (`aggregationType`)

- `union` : Union des résultats de l'ensemble des modèles appelés.
- `intersection` / `intersection2models` : Consensus retenant uniquement les vedettes identifiées par plusieurs modèles indépendants.
- `cross` : Réordonnancement (_reranking_) par Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`).
- `llm` : Exclusion des propositions hors-sujet via prompt ciblé vers un LLM distant (_Llama 3.1_).

---

## 📡 Spécification de l'API

### 1. `GET /subject_indexation/` (Point d'entrée principal)

Génère les propositions de vedettes RAMEAU.

#### Paramètres de requête (Query Parameters) :

| Paramètre          | Type  | Défaut     | Obligatoire | Description                                                                                                           |
| :----------------- | :---- | :--------- | :---------: | :-------------------------------------------------------------------------------------------------------------------- |
| `Title`            | `str` | -          |   **Oui**   | Titre du document à indexer.                                                                                          |
| `Agent`            | `str` | -          |   **Oui**   | Numéro RCR (9 chiffres) de la bibliothèque demandeuse (contrôlé via liste blanche des établissements testeurs).       |
| `Summary`          | `str` | `""`       |     Non     | Résumé ou extrait du document (contexte additionnel pour l'IA).                                                       |
| `docId`            | `str` | `""`       |     Non     | Identifiant PPN du document.                                                                                          |
| `models`           | `str` | `None`     |     Non     | Modèles à interroger, séparés par virgules (ex. `victor1_concept,victor3_chain` ou `*`). Par défaut : modèles _best_. |
| `aggregationType`  | `str` | `None`     |     Non     | Stratégie(s) d'agrégation (ex. `cross`, `union`, `llm`, `intersection2models`).                                       |
| `subjectsMaxCount` | `int` | `5`        |     Non     | Nombre maximal de propositions retournées par modèle.                                                                 |
| `vocabulary`       | `str` | `'rameau'` |     Non     | Référentiel contrôlé ciblé (défaut : `rameau`).                                                                       |
| `Format`           | `str` | `'json'`   |     Non     | Format de la réponse : `json`, `text`, `text_flat`, `text_tree`, `html`, `html_flat`, `html_tree`.                    |

### 2. `GET /rameau`

Interface HTML légère permettant de tester interactivement l'indexation avec formulaire de saisie (RCR, Titre, Résumé, Modèles).

### 3. `GET /health`

Sonde de santé (_Healthcheck_) pour Docker et la supervision ABES.

- Retourne : `{"status": "ok"}` (HTTP 200).

---

## ⚙ Configuration (`.env`)

La configuration est centralisée dans le module [`src/config.py`](./src/config.py) et se base sur les variables d'environnement.

Pour initialiser la configuration locale :

```bash
cp .env-dist .env
```

### Paramètres disponibles :

| Variable            | Description                                          | Valeur par défaut |
| :------------------ | :--------------------------------------------------- | :---------------- |
| `IAR_LLM_URL`       | URL de l'API LLM (compatible API OpenAI / vLLM ABES) | -                 |
| `IAR_LLM_KEY`       | Clé d'API d'authentification pour le service LLM     | -                 |
| `IAR_LLM_MODEL`     | Nom du modèle LLM pour le filtrage sémantique        | -                 |
| `IAR_QDRANT_HOST`   | Hôte du serveur vectoriel Qdrant                     | `localhost`       |
| `IAR_QDRANT_PORT`   | Port d'écoute du serveur Qdrant                      | `6333`            |
| `IAR_API_HTTP_HOST` | Adresse d'écoute du serveur FastAPI                  | `0.0.0.0`         |
| `IAR_API_HTTP_PORT` | Port d'écoute de l'application                       | `8071`            |
| `IAR_CORS_ORIGINS`  | Origines autorisées pour les requêtes CORS           | `*`               |
| `ENABLE_GPU`        | Activation du GPU CUDA pour l'inférence Torch        | `true`            |

> ⚠️ **Sécurité** : Ne jamais commiter le fichier `.env` contenant les clés d'API réelles.

---

## 📦 Installation & Démarrage

### 1. En environnement local (Python)

#### Prérequis :

- Python 3.10+
- Compilateur C++ (`build-essential`) pour la compilation native de la bibliothèque Omikuji
- Une instance Qdrant en fonctionnement sur le port `6333`

#### Étapes :

```bash
# 1. Création et activation de l'environnement virtuel
python -m venv venv
# Linux / macOS :
source venv/bin/activate
# Windows :
venv\Scripts\activate

# 2. Installation des dépendances
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126  # Support CUDA si GPU disponible
pip install -r requirements.txt

# 3. Préparation des dictionnaires linguistiques NLTK
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 4. Lancement du serveur FastAPI en mode rechargement automatique
python -m uvicorn src.rameau:app --host 0.0.0.0 --port 8071 --reload
```

### 2. Déploiement avec Docker

Le [`Dockerfile`](./Dockerfile) multi-stage configure l'image de production avec utilisateur non-root (`appuser`), support GPU CUDA et sonde de santé.

```bash
# Construction de l'image
docker build -t abesesr/iar:develop-api --target api-image .

# Lancement du conteneur
docker run -d \
  --name iar-api \
  --restart unless-stopped \
  -p 8071:8071 \
  --env-file .env \
  -v $(pwd)/volumes/csv:/app/data/csv:ro \
  abesesr/iar:develop-api
```

---

## 📊 Supervision & Logs

- **Vérification de l'état du conteneur** :
  ```bash
  docker ps -f name=iar-api
  ```
- **Affichage des logs en temps réel** :
  ```bash
  docker logs -f --tail=100 iar-api
  ```
- **Supervision centralisée ABES** :
  - Consultation via **Dozzle** sur les serveurs de plateforme (ex. port `29999` sur `donut-test` ou `donut-prod`).
  - Centralisation des logs applicatifs dans le puits de logs Kibana de l'ABES.
  - Healthcheck HTTP : `curl -f http://localhost:8071/health`

---

## 💾 Sauvegarde & Restauration

Conformément aux pratiques d'exploitation de l'ABES :

### 1. Éléments à Sauvegarder

- **Configuration** : le fichier `/opt/pod/iar-api/.env`.
- **Référentiels RAMEAU (Volumes CSV)** : répertoire [`volumes/csv/`](./volumes/csv/) contenant :
  - `rameau_ancestors_df.csv`
  - `rameau_parents.csv`
  - `rameau_subdivisionsONLY.csv`
- **Base Vectorielle Qdrant** : snapshots des collections vectorielles (`concepts_allMin_only_mono`, `concepts_distiluse_only_mono`, `concepts_e5-large_only_mono`).
  ```bash
  # Déclenchement d'un snapshot Qdrant via l'API REST
  curl -X POST "http://localhost:6333/collections/concepts_allMin_only_mono/snapshots"
  ```

### 2. Procédure de Restauration

1. Cloner la version ciblée du dépôt sur le serveur d'exploitation.
2. Restaurer le fichier `.env` depuis les sauvegardes sécurisées de l'ABES (ex. serveur `socorro`/`sotora`).
3. Restaurer les volumes CSV dans `volumes/csv/`.
4. Pour Qdrant, réinjecter les snapshots via l'API :
   ```bash
   curl -X PUT -F "snapshot=@snapshot_concepts.snapshot" \
     "http://localhost:6333/collections/concepts_allMin_only_mono/snapshots/upload"
   ```
5. Redémarrer le conteneur applicatif :
   ```bash
   docker restart iar-api
   ```

---

## 📁 Structure du Projet & Modules

- **[`src/rameau.py`](./src/rameau.py)** : Point d'entrée principal FastAPI. Orchestre les routes, la validation des RCR, l'interrogation des modèles d'embedding et le formatage Unimarc 606.
- **[`src/config.py`](./src/config.py)** : Chargement et validation typée des variables d'environnement.
- **[`src/embed_lib.py`](./src/embed_lib.py)** : Moteur de recherche vectorielle Qdrant, prétraitements linguistiques (lemmatisation Simplemma, NLTK).
- **[`src/omk.py`](./src/omk.py)** : Classifieur extrême multi-label Omikuji couplé au vectoriseur TF-IDF pour les vedettes RAMEAU.
- **[`volumes/csv/`](./volumes/csv/)** : Référentiels hiérarchiques et règles de subdivision RAMEAU.
- **[`test/`](./test/)** : Outils d'évaluation sémantique et de non-régression.

---

## 🧪 Tests & Évaluation

Le répertoire [`test/`](./test/) regroupe la suite de tests et de benchmark sémantique :

### 1. Test d'intégration vectorielle

Vérifie la connectivité bout en bout avec Qdrant et l'inférence du modèle d'embedding :

```bash
python test/test_qdrant_integration.py
```

### 2. Évaluation de la précision sémantique

Calcule la justesse métier (Top-k accuracy) des prédictions sur un corpus de notices annotées :

```bash
# Exécution scriptée
python test/test_classification.py

# Ou via notebook interactif
jupyter notebook test/test_classification.ipynb
```

### 3. Données de Test ([`test/data/`](./test/data/))

- [`test_data.csv`](./test/data/test_data.csv) : Liste de PPN isolés réservés à l'évaluation (hors apprentissage).
- [`test_rameau_export.csv`](./test/data/test_rameau_export.csv) : Données de vérité terrain pour le calcul des scores de précision.
