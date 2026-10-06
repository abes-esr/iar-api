# 🚀 iar-api : Service Web d'Indexation Sujet RAMEAU

[![Docker Pulls](https://img.shields.io/docker/pulls/abesesr/iar.svg)](https://hub.docker.com/r/abesesr/iar/)
[![Buildx Publish](https://github.com/abes-esr/iar-api/actions/workflows/buildx-pubtodockerhub.yml/badge.svg)](https://github.com/abes-esr/iar-api/actions/workflows/buildx-pubtodockerhub.yml)

---

## 📌 Sommaire

- [1. Description brève du projet](#1-description-brève-du-projet)
- [2. Liens vers les autres parties du projet](#2-liens-vers-les-autres-parties-du-projet)
- [3. Flux entrants / sortants](#3-flux-entrants--sortants)
- [4. Endpoints & Explication des arguments](#4-endpoints--explication-des-arguments)
- [5. Explications des variables d'environnement](#5-explications-des-variables-denvironnement)
- [6. Diagramme d'architecture](#6-diagramme-darchitecture)
- [7. Procédure de déploiement / d'installation](#7-procédure-de-déploiement--dinstallation)
- [8. Procédure de supervision](#8-procédure-de-supervision)
- [9. Procédure de restauration](#9-procédure-de-restauration)
- [10. Procédure de testing](#10-procédure-de-testing)

---

## 1. Description brève du projet

**`iar-api`** est le service web backend du projet **IAR** (**I**ndexation **A**utomatique **R**AMEAU), conçu et maintenu par l'**ABES** (Agence Bibliographique de l'Enseignement Supérieur).

Développé avec **FastAPI**, ce composant propose une API REST haute performance qui suggère automatiquement des vedettes-matières **RAMEAU** Unimarc (zone `606`) pour l'indexation de notices bibliographiques (titre, résumé, PPN). Il combine la recherche sémantique vectorielle dense via **Qdrant**, l'extraction multi-modèles d'embeddings Sentence-Transformers (_Victor 1, 2, 3_), le réordonnancement par Cross-Encoder et le filtrage des faux-positifs par Modèle de Langage (_LLM Llama 3.1_).

---

## 2. Liens vers les autres parties du projet

Le projet IAR est composé de plusieurs modules interdépendants hébergés sur l'organisation [abes-esr](https://github.com/abes-esr) :

| Dépôt GitHub                                                           | Rôle & Description                                                                                                                |
| :--------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| [**iar-docker**](https://github.com/abes-esr/iar-docker)               | Configuration Docker Compose pour le déploiement complet des conteneurs de la plateforme (API, Qdrant, Ollama/vLLM, Dozzle).      |
| [**iar-api**](https://github.com/abes-esr/iar-api)                     | Ce dépôt : API FastAPI exposant le service de suggestion d'indexation sujet.                                                      |
| [**iar-vectorisation**](https://github.com/abes-esr/iar-vectorisation) | Scripts de génération des vecteurs d'embeddings RAMEAU et de peuplement des collections Qdrant.                                   |
| [**iar-batch-docker**](https://github.com/abes-esr/iar-batch-docker)   | Déploiement Docker pour les traitements par lots et les interfaces de mise à jour des données.                                    |
| [**iar-batch-dump**](https://github.com/abes-esr/iar-batch-dump)       | Extraction, transformation et génération des dumps de données d'autorités RAMEAU et notices Sudoc.                                |
| [**iar-script-winibw**](https://github.com/abes-esr/iar-script-winibw) | Script client (VBScript) s'intégrant au logiciel de catalogage **WinIBW** pour interroger l'API depuis le poste des catalogueurs. |

---

## 3. Flux entrants / sortants

### Flux entrants :

- **Requêtes HTTP GET** depuis :
  - Les clients de catalogage **WinIBW** (via [iar-script-winibw](https://github.com/abes-esr/iar-script-winibw)).
  - Des applications web ou scripts tiers de l'ABES (Sudoc, interfaces d'indexation).
  - La sonde de supervision HTTP (requêtes périodiques sur `/health`).
- **Fichiers de données (Volumes locaux / montages)** :
  - Fichiers d'autorités RAMEAU dans [`volumes/csv/`](./volumes/csv/) générés par [iar-batch-dump](https://github.com/abes-esr/iar-batch-dump) : `rameau_ancestors_df.csv`, `rameau_parents.csv`, `rameau_subdivisionsONLY.csv`.

### Flux sortants :

- **Réponses HTTP REST** : Flux de vedettes RAMEAU Unimarc 606 au format JSON structuré ou texte/HTML plat ou hiérarchique.
- **Requêtes gRPC / REST vers Qdrant** (port `6333`) : Recherche de similarité par cosinus sur les collections de vecteurs.
- **Requêtes HTTP REST vers le serveur LLM** (compatible OpenAI API, ex. vLLM / Ollama sur `IAR_LLM_URL`) : Filtrage sémantique des propositions.
- **Flux de logs (stdout/stderr)** : Traitement et collecte centralisée des logs applicatifs par Dozzle et le puits de logs Kibana de l'ABES.

---

## 4. Endpoints & Explication des arguments

### 1. `GET /subject_indexation/` (Indexation sémantique)

Génère la liste ordonnée des vedettes-matières RAMEAU suggérées pour une notice.

| Paramètre          | Type  | Défaut     | Obligatoire | Description                                                                                                                                                                                      |
| :----------------- | :---- | :--------- | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Title`            | `str` | -          |   **Oui**   | Titre du document à indexer.                                                                                                                                                                     |
| `Agent`            | `str` | -          |   **Oui**   | Numéro RCR (9 chiffres) ou identifiant de l'établissement demandeur. Contrôlé via la liste blanche des établissements testeurs.                                                                  |
| `Summary`          | `str` | `""`       |     Non     | Résumé ou texte contextuel du document.                                                                                                                                                          |
| `docId`            | `str` | `""`       |     Non     | Identifiant du document (ex. PPN de la notice Sudoc).                                                                                                                                            |
| `models`           | `str` | `None`     |     Non     | Modèles à interroger, séparés par virgules : `victor1_concept`, `victor1_chain`, `victor2`, `victor3_chain`, `victor3_chain_en`, ou `*` pour tous. Par défaut : `victor1_concept,victor3_chain`. |
| `aggregationType`  | `str` | `None`     |     Non     | Stratégie(s) d'agrégation séparées par virgules : `union`, `intersection`, `intersection2models`, `intersection2models1best`, `cross`, `llm`, `lilimarlene`, `qdrant`, `embbed`.                 |
| `subjectsMaxCount` | `int` | `5`        |     Non     | Nombre maximum de suggestions retenues par modèle.                                                                                                                                               |
| `vocabulary`       | `str` | `'rameau'` |     Non     | Vocabulaire contrôlé cible (défaut : `rameau`).                                                                                                                                                  |
| `Format`           | `str` | `'json'`   |     Non     | Format de la réponse : `json`, `text`, `text_flat`, `text_tree`, `html`, `html_flat`, `html_tree`.                                                                                               |

#### Exemples de requêtes complètes selon l'environnement :

- **Test** :
  `http://donut-test.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=union,cross,qdrant&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree`
- **Production** :
  `http://donut-prod.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=union,cross,qdrant&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree`

### 2. `GET /rameau` (Interface de test HTML)

Interface web interactive pour tester les suggestions d'indexation directement dans un navigateur :

- **Test** : `http://donut-test.abes.fr:8071/rameau`
- **Production** : `http://donut-prod.abes.fr:8071/rameau`

### 3. `GET /health` (Sonde de vie)

Sonde de supervision utilisée par Docker et les outils de surveillance.

- **Réponse HTTP 200** : `{"status": "ok"}`

### 4. `GET /docs` & `GET /redoc` (OpenAPI)

Documentation interactive Swagger UI et ReDoc générée automatiquement par FastAPI :

- **Test** : `http://donut-test.abes.fr:8071/docs`
- **Production** : `http://donut-prod.abes.fr:8071/docs`

---

## 5. Explications des variables d'environnement

Les variables d'environnement sont gérées de manière centralisée par [`src/config.py`](./src/config.py) et définies dans le fichier `.env` (dérivé de [`.env-dist`](./.env-dist)).

| Variable            | Description                                                                       | Exemple / Défaut                           |
| :------------------ | :-------------------------------------------------------------------------------- | :----------------------------------------- |
| `IAR_LLM_URL`       | URL de base de l'API LLM (compatible OpenAI API) pour le filtrage sémantique.     | `localhost` (ou `http://iar-llm:11434/v1`) |
| `IAR_LLM_KEY`       | Clé d'API secrète pour s'authentifier auprès du service LLM.                      | -                                          |
| `IAR_LLM_MODEL`     | Nom du modèle LLM à utiliser pour éliminer les mots-clés intrus.                  | `llama-3.1-8b`                             |
| `IAR_QDRANT_HOST`   | Hôte du serveur de base de données vectorielle Qdrant.                            | `localhost` (ou `iar-qdrant`)              |
| `IAR_QDRANT_PORT`   | Port d'écoute de l'instance Qdrant.                                               | `6333`                                     |
| `IAR_API_HTTP_HOST` | Adresse d'écoute du serveur FastAPI / Uvicorn.                                    | `0.0.0.0`                                  |
| `IAR_API_HTTP_PORT` | Port d'écoute de l'API.                                                           | `8071`                                     |
| `IAR_CORS_ORIGINS`  | Liste des domaines autorisés pour les requêtes Cross-Origin (CORS).               | `*`                                        |
| `ENABLE_GPU`        | Active l'accélération matérielle CUDA si une carte GPU compatible est disponible. | `true`                                     |

> 🔒 **Consigne de sécurité** : Ne jamais versionner le fichier `.env` contenant les identifiants ou clés de production.

---

## 6. Diagramme d'architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         CLIENTS & IHM                                           │
│   • WinIBW (Catalogueurs Sudoc via iar-script-winibw)                                           │
│   • Applications Web & Navigateurs (/rameau ou API externe)                                     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │ Requête HTTP GET /subject_indexation/
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  iar-api (FastAPI : Port 8071)                                  │
│                                                                                                 │
│  1. ROUTEUR & FILTRES :                                                                         │
│     • Validation RCR (liste blanche des établissements testeurs)                                │
│     • Détection automatique de langue (Langdetect -> Français / Multilingue)                    │
│                                                                                                 │
│  2. EXTRACTION VECTORIELLE (Sentence-Transformers) :                                            │
│     ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────────────┐  │
│     │   Victor 1 (all-Mini)  │  │   Victor 2 (distiluse) │  │    Victor 3 (e5-large)         │  │
│     │   • Concepts RAMEAU    │  │   • Multilingue        │  │    • Multilingue FR / EN       │  │
│     └───────────┬────────────┘  └───────────┬────────────┘  └───────────────┬────────────────┘  │
│                 │                           │                               │                   │
│                 └───────────────────────────┼───────────────────────────────┘                   │
│                                             │ Recherche K-PPV (Similarité Cosinus)              │
│                                             ▼                                                   │
│  3. POST-TRAITEMENTS & AGRÉGATION :                                                             │
│     • Reranking par Cross-Encoder (mmarco-mMiniLMv2)                                            │
│     • Consensus / Intersections de modèles (union, cross, intersection2models)                  │
│     • Structuration hiérarchique RAMEAU (Ancestors / Parents / Subdivisions)                    │
│     • Filtrage sémantique des intrus via LLM                                                    │
│                                             │                                                   │
│  4. FORMATAGE DE SORTIE (Unimarc 606) :     │                                                   │
│     • Formats disponibles : JSON structuré, Text (Flat/Tree), HTML (Flat/Tree)                  │
└───────────────▲─────────────────────────────┼─────────────────────────────┬─────────────────────┘
                │                             │                             │
                │ Requêtes                    │ Requêtes                    │ Relations
                │ Vectorielles                │ Filtrage LLM                │ Hiérarchiques
                ▼                             ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────┐ ┌─────────────────────────────────────┐
│      Qdrant Vector DB         │ │      Service LLM      │ │             Volumes CSV             │
│        (Port 6333)            │ │ (Port 11434 / distant)│ │          (./volumes/csv/)           │
│  • concepts_allMin_only_mono  │ │ • Llama 3.1           │ │  • rameau_ancestors_df.csv          │
│  • concepts_distiluse_only... │ │ • Filtrage sémantique │ │  • rameau_parents.csv               │
│  • concepts_e5-large_only_mono│ │   des faux-positifs   │ │  • rameau_subdivisionsONLY.csv      │
└───────────────────────────────┘ └───────────────────────┘ └─────────────────────────────────────┘
                ▲                                                           ▲
                │ Logs stdout                                               │ Métriques & Alertes
                └─────────────────────────────┬─────────────────────────────┘
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SUPERVISION & OBSERVABILITÉ                                   │
│   • Dozzle (Port 29999) : Consultation des logs conteneurs en temps réel                        │
│   • Grafana / Kibana (Port 3000) : Puits de logs centralisé ABES (diplotaxis7-*)                │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Procédure de déploiement / d'installation

> ℹ️ **Architecture de déploiement ABES** : Sur les serveurs de test et de production (`donut-test` et `donut-prod`), seul le dépôt [**iar-docker**](https://github.com/abes-esr/iar-docker) est installé (dans `/opt/pod/iar-docker/`). Il instancie l'image Docker précompilée de l'API (`abesesr/iar:*-api`). Les procédures ci-dessous décrivent le build local ou l'exécution autonome du code source pour les développeurs.

### A. Déploiement avec Docker (Recommandé)

Le projet s'exécute de façon conteneurisée via le [`Dockerfile`](./Dockerfile) de l'API ou au sein de la pile globale [iar-docker](https://github.com/abes-esr/iar-docker).

```bash
# 1. Cloner le dépôt sur la machine hôte
git clone https://github.com/abes-esr/iar-api.git
cd iar-api

# 2. Copier le fichier d'environnement
cp .env-dist .env
# 3. Éditer le fichier .env selon l'environnement ciblé (localhost, test ou prod)

# 4. Construire l'image Docker de production
docker build -t abesesr/iar:develop-api --target api-image .

# 5. Lancer le conteneur avec montage des volumes de données CSV
docker run -d \
  --name iar-api \
  --restart unless-stopped \
  -p 8071:8071 \
  --env-file .env \
  -v $(pwd)/volumes/csv:/app/data/csv:ro \
  abesesr/iar:develop-api
```

### B. Installation locale pour le développement

```bash
# 1. Création et activation de l'environnement virtuel Python
python -m venv venv
# Linux / macOS :
source venv/bin/activate
# Windows :
venv\Scripts\activate

# 2. Installation des dépendances
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126  # Support CUDA si GPU disponible
pip install -r requirements.txt

# 3. Téléchargement des ressources NLTK
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# 4. Initialisation du fichier .env
cp .env-dist .env
# 5. Éditer le fichier .env selon l'environnement ciblé (localhost, test ou prod)

# 6. Démarrage de l'API avec rechargement à chaud
python -m uvicorn src.rameau:app --host 0.0.0.0 --port 8071 --reload
```

---

## 8. Procédure de supervision

### 1. Contrôle de l'état du conteneur et des logs

```bash
# Vérifier l'état du conteneur
docker ps -f name=iar-api

# Visualiser les 100 dernières lignes de logs en direct
docker logs -f --tail=100 iar-api
```

### 2. Sonde de santé applicative (Healthcheck)

L'état de fonctionnement de l'API est interrogé via son endpoint dédié :

```bash
curl -f http://localhost:8071/health
# Réponse attendue : {"status":"ok"}
```

### 3. Consultation visuelle des logs via Dozzle

Les logs de l'API et de ses dépendances sont consultables en continu via Dozzle :

- **Test** : `http://donut-test.abes.fr:29999/`
- **Production** : `http://donut-prod.abes.fr:29999/`

### 4. Supervision globale et observabilité

Les métriques systèmes et applicatives sont agrégées sur les serveurs Grafana de l'ABES :

- `http://diplotaxis7-dev.v212.abes.fr:3000`
- `http://diplotaxis7-test.v202.abes.fr:3000`
- `http://diplotaxis7-prod.v102.abes.fr:3000`

---

## 9. Procédure de restauration

En cas d'incident sur l'infrastructure ou de réinstallation sur un nouveau serveur (`donut-test` ou `donut-prod`), la procédure de restauration s'applique à l'environnement [**iar-docker**](https://github.com/abes-esr/iar-docker) (`/opt/pod/iar-docker/`), seul dossier présent sur les serveurs hôtes.

### Étape 1 : Récupération globale via `rsync`

Les sauvegardes automatiques de l'ABES effectuant une sauvegarde globale de l'ensemble du serveur sur les machines de stockage dédiées (`socorro.abes.fr` / `sotora.abes.fr`), une unique commande `rsync` permet de restaurer rapidement l'intégralité du répertoire de déploiement `iar-docker` (incluant le fichier `.env`, les volumes de référentiels `volumes/csv/` et les configurations) :

```bash
# Restauration globale du répertoire iar-docker (adapter le serveur source : donut-test ou donut-prod)
rsync -avzP socorro.abes.fr:/backup/donut-prod/opt/pod/iar-docker/ /opt/pod/iar-docker/
```

### Étape 2 : Restauration des collections vectorielles Qdrant

Si la base Qdrant doit être réinitialisée, injecter les snapshots via l'API REST de Qdrant :

```bash
# Restauration de la collection principale des concepts
curl -X PUT -F "snapshot=@/opt/pod/iar-docker/volumes/qdrant/snapshots/concepts_allMin_only_mono.snapshot" \
  "http://localhost:6333/collections/concepts_allMin_only_mono/snapshots/upload"

# Restauration de la collection multilingue
curl -X PUT -F "snapshot=@/opt/pod/iar-docker/volumes/qdrant/snapshots/concepts_e5-large_only_mono.snapshot" \
  "http://localhost:6333/collections/concepts_e5-large_only_mono/snapshots/upload"
```

Les collections restaurées sont consultables immédiatement sur le tableau de bord Qdrant :

- **Local** : `http://localhost:6333/dashboard#/collections`
- **Test** : `http://donut-test.abes.fr:6333/dashboard#/collections`
- **Production** : `http://donut-prod.abes.fr:6333/dashboard#/collections`

### Étape 3 : Redémarrage et vérification

```bash
# Se placer dans le répertoire d'exploitation
cd /opt/pod/iar-docker/

# Démarrer l'ensemble des conteneurs
docker compose up -d

# Valider le bon démarrage
curl -f http://localhost:8071/health
```

---

## 10. Procédure de testing

La suite de tests et de validation s'articule autour du répertoire [`test/`](./test/) :

### 1. Test d'intégration vectorielle Qdrant

Ce test valide la chaîne complète d'embedding et d'interrogation de la base vectorielle locale :

```bash
python test/test_qdrant_integration.py
```

### 2. Évaluation de la justesse sémantique (Accuracy & Benchmark)

Les scripts et notebooks évaluent l'exactitude des suggestions d'autorités RAMEAU (Top-1, Top-3, Top-5 accuracy) sur un échantillon représentatif de notices du Sudoc afin de prévenir toute régression algorithmique :

```bash
# Exécution scriptée autonome
python test/test_classification.py

# Version enrichie v2
python test/test_classification_v2.py

# Exécution interactive via Jupyter Notebook
jupyter notebook test/test_classification.ipynb
```

### 3. Jeux de données de test ([`test/data/`](./test/data/))

- [`test/data/test_data.csv`](./test/data/test_data.csv) : Liste de PPN isolés réservés à l'évaluation pour éviter les biais d'apprentissage.
- [`test/data/test_rameau_export.csv`](./test/data/test_rameau_export.csv) : Dataset de référence annoté (Titres, Résumés, Vedettes RAMEAU réelles attendues).
