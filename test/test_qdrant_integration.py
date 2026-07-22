"""
Script de test d'intégration pour valider la recherche vectorielle avec Qdrant.

Ce test réutilise la bibliothèque interne `src.embed_lib` pour s'assurer
de la non-régression du prétraitement, de la vectorisation et de la recherche.
"""

import sys
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd

# Ajout du répertoire parent au path pour importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embed_lib import embedding_qdrant

def run_integration_test(titre: str, resume: str, host: str = "localhost", port: int = 6333, collection: str = "rameau_concepts"):
    """
    Exécute un test d'intégration de recherche vectorielle Qdrant.

    Args:
        titre (str): Le titre du document à indexer.
        resume (str): Le résumé du document.
        host (str): L'hôte de la base vectorielle Qdrant.
        port (int): Le port de la base vectorielle Qdrant.
        collection (str): Le nom de la collection à interroger.

    Returns:
        pd.DataFrame: Les propositions de vedettes RAMEAU.
    """
    print(f"Connexion à Qdrant sur {host}:{port}...")
    qdrant_client = QdrantClient(host=host, port=port)
    
    print("Chargement du modèle SentenceTransformer (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"Lancement de la prédiction sur la collection '{collection}'...")
    results = embedding_qdrant(
        titre=titre,
        resume=resume,
        qdrant_client=qdrant_client,
        encoder=encoder,
        collection_name=collection,
        subjects_max_count=6
    )
    
    return results

if __name__ == "__main__":
    # Paramètres de test par défaut
    test_titre = "Le rôle des bibliothèques publiques dans l'accès à l'information"
    test_resume = "Une étude de cas sur la numérisation des fonds documentaires et l'inclusion numérique."
    
    try:
        df_res = run_integration_test(
            titre=test_titre,
            resume=test_resume,
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            collection=os.getenv("QDRANT_COLLECTION", "rameau_concepts")
        )
        print("\nRésultats de la prédiction :")
        print(df_res)
    except Exception as e:
        print(f"\n[ERREUR] Échec du test d'intégration : {e}", file=sys.stderr)
        print("Note : Assurez-vous que le serveur Qdrant est démarré et accessible.", file=sys.stderr)



