# 🚀 Service Web d'Indexation Sujet RAMEAU (API FastAPI)

Ce composant backend constitue le cœur du projet **RAMEAU** de l'ABES (Agence Bibliographique de l'Enseignement Supérieur). Il fournit un service web REST (exposé via **FastAPI**) permettant de suggérer des vedettes-matières **RAMEAU** Unimarc (zone `606`) enrichies par de l'intelligence artificielle à partir des métadonnées d'un document (titre, résumé, identifiant PPN).

---

## 📌 Sommaire

- [Vue d'ensemble & Architecture](#-vue-densemble--architecture)
- [Fonctionnalités Principales](#-fonctionnalités-principales)
- [Modèles d'Embedding & Stratégies d'Agrégation](#-modèles-dembedding--stratégies-dagrégation)
- [Ajustements Sécurité & Corrections Appliquées](#-ajustements-sécurité--corrections-appliquées)
- [Spécification de l'API](#-spécification-de-lapi)
- [Installation & Déploiement](#-installation--déploiement)
- [Structure du Projet & Modules `src/`](#-structure-du-projet--modules-src)
- [Tests & Évaluation](#-tests--évaluation)

---

## 🏗 Vue d'ensemble & Architecture

Le serveur reçoit les requêtes en provenance d'IHM de catalogage (ex: **WinIBW** via VBScript) ou d'interfaces Web. Il traite la notice bibliographique via une chaîne d'analyse à plusieurs étages :

```
┌─────────────────────────────────────────────────────────────┐
│                       Notice Unimarc                        │
│                  (Titre, Résumé, PPN, RCR)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Request GET /subject_indexation/
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend Router                   │
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
│              Recherche Vectorielle (Qdrant / FAISS)         │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Module de Post-Traitements & Agrégation          │
│   • Intersections & Reranking Cross-Encoder                 │
│   • Filtrage/Sélection LLM (Llama 3.1 via Ollama/vLLM)      │
│   • Structuration Hiérarchique RAMEAU (Arborescence/Roots)  │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Format de Sortie (Unimarc 606)                  │
│       Format JSON, HTML (flat/tree) ou Text (flat/tree)     │
└──────────────────────────────┬──────────────────────────────┘
```

---

## ✨ Fonctionnalités Principales

1. **Recherche Vectorielle sémantique hybride :** Supporte aussi bien **Qdrant** (base de données vectorielle) que **FAISS** pour l'indexation ultra-rapide des concepts et chaînes d'autorités RAMEAU.
2. **Détection Automatique de Langue :** Si le titre ou le résumé n'est pas en français (`langdetect`), le serveur bascule automatiquement sur des modèles d'embedding multilingues (`victor3_chain_en`).
3. **Agrégation Intelligente (Consensus & LLM) :** Combine les prédictions de plusieurs modèles par intersection, reranking par Cross-Encoder (`mmarco-mMiniLMv2-L12-H384-v1`), ou exclusion des intrus via un LLM (_Llama 3.1_).
4. **Mise en forme Unimarc 606 automatisée :**
   - Génération de lignes `606 ##$3<PPN>$a<Libellé>$2rameau`.
   - Prise en compte des règles spécifiques de subdivision (`rameau_subdivisionsONLY.csv`).
   - Structuration arborescente (relation parent/enfant via `rameau_parents.csv` et `rameau_ancestors_df.csv`).
5. **Restriction par RCR / Établissements Autorisés :** Validation de l'identifiant `Agent` (RCR) par rapport à une liste blanche d'établissements testeurs.

---

## 🤖 Modèles d'Embedding & Stratégies d'Agrégation

### Modèles IA (_Victor_)

- **`victor1_concept` / `victor1_chain` :** Basé sur `all-MiniLM-L6-v2`.
- **`victor2` :** Basé sur `distiluse-base-multilingual-cased-v2`.
- **`victor3_chain` / `victor3_chain_en` :** Basé sur `intfloat/multilingual-e5-large`.

### Stratégies d'Agrégation (`aggregationType`)

- `union` : Union simple des résultats de tous les modèles interrogés.
- `intersection` : Conservation uniquement des sujets suggérés par _tous_ les modèles.
- `intersection2models` : Conservation des sujets suggérés par _au moins 2 modèles_.
- `llm` : Le modèle LLM (_Llama 3.1_) reçoit le résumé et la liste des propositions, puis élimine les mots-clés non pertinents.
- `cross` : Re-classement des propositions (_Reranking_) grâce à un Cross-Encoder.
- `embbed` : Re-classement des propositions via mesure de similarité cosinus.

---

## 📡 Spécification de l'API

### Endpoint principal : `GET /subject_indexation/`

#### Paramètres de requête (Query Parameters) :

| Paramètre          | Type  | Défaut     | Description                                                                                     |
| :----------------- | :---- | :--------- | :---------------------------------------------------------------------------------------------- |
| `Title`            | `str` | _Requis_   | Titre principal du document.                                                                    |
| `Summary`          | `str` | `""`       | Résumé ou extrait du document (servant de contexte à l'IA).                                     |
| `docId`            | `str` | `""`       | Identifiant PPN du document à indexer.                                                          |
| `models`           | `str` | `None`     | Modèles à utiliser séparés par des virgules (ex: `victor1_concept,victor3_chain` ou `*`).       |
| `aggregationType`  | `str` | `None`     | Méthode d'agrégation (ex: `intersection2models,llm`, `cross`, etc.).                            |
| `vocabulary`       | `str` | `'rameau'` | Vocabulaire d'autorité visé.                                                                    |
| `subjectsMaxCount` | `int` | `5`        | Nombre maximal de sujets retournés par modèle.                                                  |
| `Agent`            | `str` | _Requis_   | Numéro RCR / Identifiant de l'établissement demandeur.                                          |
| `Format`           | `str` | `'json'`   | Format de réponse : `json`, `text`, `text_flat`, `text_tree`, `html`, `html_flat`, `html_tree`. |

---

## 📦 Installation & Déploiement

### 1. Prérequis

- Python 3.10+
- Une instance **Qdrant** en local (Port `6333`) ou l'accès aux index FAISS (`.faiss`).
- Accès à un serveur LLM compatible OpenAI API (ex: **Ollama**, **vLLM** ou vLLM ABES).

### 2. Installation via `requirements.txt`

Cloner le dépôt et installer les dépendances Python spécifiées dans le fichier `requirements.txt` :

```bash
# Création d'un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Linux/macOS
# venv/Scripts/activate  # Sur Windows

# Installation des paquets
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lancement du serveur Uvicorn

```bash
python -m uvicorn rameau:app --host 0.0.0.0 --port 8071 --reload
```

L'interface de documentation interactive (Swagger UI) sera disponible sur : `http://localhost:8071/docs`.

---

## 📁 Structure du Projet & Modules `src/`

- **`src/rameau.py`** : Point d'entrée principal du serveur Web FastAPI. Définit la route `/subject_indexation/` et orchestre les modèles d'embedding, les agrégations (consensus, cross-encoder, LLM) et la mise en forme Unimarc 606.
- **`src/embed_lib.py`** : Module de recherche vectorielle. Fournit la chaîne de nettoyage linguistique native (`unicodedata`, `re`, `simplemma`, `nltk`) et les fonctions d'embedding `embedding_faiss` et `embedding_qdrant`.
- **`src/omk.py`** : Module de prédiction multi-label extrême. Intègre la classification `Omikuji` couplée au vectoriseur TF-IDF et décodage de vedettes RAMEAU.

---

## 🧪 Tests & Évaluation

Le dossier [test/](test/) contient les outils d'évaluation de la performance du modèle et de non-régression de l'API.

### 1. Test d'intégration vectorielle (`test_qdrant_integration.py`)

Ce script valide l'intégration de bout en bout de l'API de recherche vectorielle sémantique en se connectant à une instance Qdrant locale. Il utilise le modèle d'embedding par défaut et interroge la collection configurée pour retourner des propositions RAMEAU.

**Exécution :**

```bash
python test/test_qdrant_integration.py
```

### 2. Évaluation de la Justesse sémantique (`test_classification.ipynb` & `test_classification_v2.ipynb`)

Ces notebooks Jupyter contiennent le pipeline complet pour évaluer la justesse algorithmique des prédictions RAMEAU. Ils calculent l'exactitude des prédictions (ex: Top-k accuracy) en utilisant des datasets de test dédiés afin de garantir la non-régression de la qualité métier lors des changements de configuration.

**Exécution :**
Ouvrir les notebooks via Jupyter Lab/Notebook ou VS Code et lancer l'exécution séquentielle des cellules :

```bash
# Lancement de Jupyter
jupyter notebook test/test_classification.ipynb
```

### 3. Données de Test (`test/data/`)

- [test_data.csv](test/data/test_data.csv) : Liste des identifiants (PPN) de notices réservés exclusivement à l'évaluation (pour éviter le biais d'entraînement).
- [test_rameau_export.csv](test/data/test_rameau_export.csv) : Corpus réduit de notices de test (PPN, Titres, vedettes RAMEAU attendues) servant de benchmark pour évaluer la justesse des prédictions.
