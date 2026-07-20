"""
Module de prédiction Omikuji RAMEAU.

Ce module permet de prédire des vedettes-matières RAMEAU grâce à la bibliothèque
de classification extrême multi-label Omikuji et une vectorisation TF-IDF.
"""

from typing import Any
import pandas as pd
import omikuji

from .embed_lib import clean_and_lemmatize


def predict_sentence(
    sentence: str,
    model: Any,
    vectorizer: Any,
    le: Any,
    densify_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Prédit les étiquettes et scores RAMEAU pour une phrase donnée via Omikuji.

    Transforme la phrase en vecteur TF-IDF, applique le modèle Omikuji
    et décode les étiquettes prédites avec le LabelEncoder fourni.

    Args:
        sentence (str): La phrase prétraitée et lemmatisée.
        model: Le modèle Omikuji chargé en mémoire.
        vectorizer: Le vectoriseur TF-IDF pré-entraîné (scikit-learn).
        le: Le LabelEncoder (scikit-learn) associant les indices aux libellés RAMEAU.
        densify_threshold (float, optional): Seuil pour densifier les poids du modèle Omikuji. Par défaut 0.05.

    Returns:
        pd.DataFrame: DataFrame contenant les colonnes ['label', 'id', 'score'].
    """
    if not sentence:
        return pd.DataFrame(columns=["label", "id", "score"])

    # Densification optionnelle des poids du modèle pour optimiser la prédiction
    model.densify_weights(densify_threshold)

    # Transformation de la phrase en matrice sparse TF-IDF
    feature_matrix = vectorizer.transform([sentence])

    # Conversion des caractéristiques TF-IDF au format attendu par Omikuji (liste de [index, valeur])
    feature_value_pairs = []
    dense_features = feature_matrix.toarray()[0]
    for idx, value in enumerate(dense_features):
        if value > 0:
            feature_value_pairs.append([idx, value])

    # Prédiction Omikuji (retourne une liste de tuples (label_idx, score))
    label_score_pairs = model.predict(feature_value_pairs)

    labels = []
    scores = []
    for label_idx, score in label_score_pairs:
        # Inverse transform du LabelEncoder
        decoded_label = le.inverse_transform([label_idx])[0]
        labels.append(decoded_label)
        scores.append(score)

    df = pd.DataFrame({"label": labels, "score": scores})

    if df.empty:
        return pd.DataFrame(columns=["label", "id", "score"])

    # Extraction séparée du PPN et du libellé (format Label#PPN)
    df["id"] = df["label"].astype(str).str.split("#").str[1]
    df["label"] = df["label"].astype(str).str.split("#").str[0]

    # Remplacement des underscores par des espaces dans les libellés
    df["label"] = df["label"].str.replace("_", " ", regex=False)

    return df[["label", "id", "score"]]


def predict_omikuji(
    titre: str,
    resume: str,
    model: Any,
    vectorizer: Any,
    le: Any,
) -> pd.DataFrame:
    """
    Prédit les vedettes-matières RAMEAU pour un titre et un résumé donnés via Omikuji.

    Prétraite et lemmatise les textes d'entrée, puis fait appel au modèle Omikuji.

    Args:
        titre (str): Le titre du document.
        resume (str): Le résumé du document.
        model: Le modèle Omikuji chargé.
        vectorizer: Le vectoriseur TF-IDF.
        le: Le LabelEncoder pour décoder les indices.

    Returns:
        pd.DataFrame: DataFrame pandas contenant les propositions ['label', 'id', 'score'].
    """
    descr_series = clean_and_lemmatize(titre, resume)
    cleaned_text = str(descr_series.iloc[0])

    return predict_sentence(cleaned_text, model, vectorizer, le)
