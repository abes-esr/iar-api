"""
Module d'indexation vectorielle RAMEAU (FAISS & Qdrant).

Ce module fournit les utilitaires de prétraitement linguistique (nettoyage natif,
lemmatisation Simplemma/NLTK) ainsi que les fonctions de recherche vectorielle
et de post-traitement des vedettes-matières RAMEAU pour les index FAISS et Qdrant.
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Any
import pandas as pd
import simplemma
import nltk

# Expression régulière pour identifier un PPN Sudoc (8 chiffres suivis de 1 chiffre ou X)
PPN_REGEX = re.compile(r"^[0-9]{8}([0-9]|X)$")


def remove_diacritics(text: str) -> str:
    """
    Supprime les diacritiques (accents) d'une chaîne de caractères sans dépendance externe.

    Args:
        text (str): La chaîne de texte brute.

    Returns:
        str: La chaîne sans accents ni diacritiques.
    """
    nfkd_form = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """
    Nettoie une chaîne de texte : passage en minuscules, suppression des diacritiques
    et normalisation des espaces superflus.

    Args:
        text (str): Le texte à nettoyer.

    Returns:
        str: Le texte nettoyé.
    """
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = remove_diacritics(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_and_lemmatize(titre: str, resume: str = "") -> pd.Series:
    """
    Nettoie et lemmatise le titre et le résumé d'un document.

    Applique une chaîne de prétraitement native (minuscules, suppression des diacritiques,
    normalisation des espaces) puis procède à une lemmatisation mot à mot en français via Simplemma.

    Args:
        titre (str): Le titre principal du document.
        resume (str, optional): Le résumé ou texte descriptif. Par défaut "".

    Returns:
        pd.Series: Série pandas contenant le texte nettoyé et lemmatisé.
    """
    text_combined = f"{titre}, {resume}" if resume else titre
    cleaned = clean_text(text_combined)

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    words = w_tokenizer.tokenize(cleaned)
    lemmatized = " ".join(simplemma.lemmatize(word, lang="fr") for word in words)

    return pd.Series([lemmatized])


def process_label(label: str) -> Tuple[str, str]:
    """
    Découpe et normalise une vedette-matière RAMEAU composée de libellés et d'identifiants PPN.

    Gère les vedettes complexes associées par '--' et les séparateurs '#'.
    Prend en compte l'inversion éventuelle (PPN#Libellé vs Libellé#PPN).

    Args:
        label (str): Chaîne brute du label (ex: "Bureaucratie#027229629" ou "027229629#Bureaucratie").

    Returns:
        Tuple[str, str]: Un tuple (libellés_découpés, ppn_découpés) joint par '--'.
    """
    if not label:
        return "", ""

    mots_ids = label.split("--")
    mots: List[str] = []
    ids: List[str] = []

    for item in mots_ids:
        if "#" in item:
            parts = item.split("#", 1)
            mots.append(parts[0])
            ids.append(parts[1])
        else:
            mots.append(item)
            ids.append("")

    # Gestion de l'inversion si le premier terme correspond au format d'un PPN
    if mots and PPN_REGEX.fullmatch(mots[0]):
        mots, ids = ids, mots

    return "--".join(mots), "--".join(ids)


def predict_qdrant(
    qdrant_client: Any,
    encoder: Any,
    text: str,
    collection: str,
    subjects_max_count: int = 5,
) -> List[Dict[str, Any]]:
    """
    Effectue une recherche vectorielle de prédiction dans une collection Qdrant.

    Args:
        qdrant_client: Le client Qdrant initialisé.
        encoder: Le modèle SentenceTransformer pour l'encodage du texte.
        text (str): Le texte d'entrée à vectoriser.
        collection (str): Nom de la collection Qdrant.
        subjects_max_count (int): Nombre maximum de résultats à retourner.

    Returns:
        List[Dict[str, Any]]: Liste de dictionnaires au format [{'score': float, 'label': Any}].
    """
    query_vector = encoder.encode(text).tolist()
    hits = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=subjects_max_count,
    )

    load_items = []
    for hit in hits:
        load_items.append({"score": hit.score, "label": hit.payload})

    return load_items






def embedding_qdrant(
    titre: str,
    resume: str,
    qdrant_client: Any,
    encoder: Any,
    collection_name: str,
    subjects_max_count: int = 5,
) -> pd.DataFrame:
    """
    Génère les propositions de vedettes RAMEAU via une collection Qdrant.

    Prétraite le texte, exécute la prédiction Qdrant, extrait et nettoie les labels
    et retourne un DataFrame structuré.

    Args:
        titre (str): Titre du document.
        resume (str): Résumé du document.
        qdrant_client: Client Qdrant.
        encoder: Modèle d'embedding.
        collection_name (str): Nom de la collection vectorielle.
        subjects_max_count (int): Limite du nombre de résultats.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes ['label', 'id', 'score'].
    """
    descr_series = clean_and_lemmatize(titre, resume)
    cleaned_text = str(descr_series.iloc[0])

    predictions = predict_qdrant(
        qdrant_client, encoder, cleaned_text, collection_name, subjects_max_count
    )

    if not predictions:
        return pd.DataFrame(columns=["label", "id", "score"])

    # Normalisation du payload Qdrant vers une étiquette textuelle
    extracted_items = []
    for item in predictions:
        lbl = item["label"]
        if isinstance(lbl, dict) and "a" in lbl:
            label_str = lbl["a"]
        else:
            label_str = str(lbl)
        extracted_items.append({"label": label_str, "score": item["score"]})

    df_res = pd.DataFrame(extracted_items)

    # Nettoyage des caractères d'échappement spécifiques RAMEAU
    df_res["label"] = (
        df_res["label"]
        .astype(str)
        .str.replace("#u#", "_", regex=False)
        .str.replace("#d#", '"', regex=False)
        .str.replace("#c#", "'", regex=False)
        .str.replace("_", " ", regex=False)
    )

    # Séparation sécurisée des libellés et des PPN
    processed_tuples = df_res["label"].apply(process_label)
    df_res["label"] = [t[0] for t in processed_tuples]
    df_res["id"] = [t[1] for t in processed_tuples]

    return df_res[["label", "id", "score"]]
