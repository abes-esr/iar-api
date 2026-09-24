import re

import pandas as pd
import simplemma
import nltk
from texthero import preprocessing as preprocessing2
import texthero as hero

def clean_and_lemmatize(titre, resume):
    
    clean_pipeline = [preprocessing2.fillna,
                      preprocessing2.lowercase,
                      preprocessing2.remove_whitespace,
                      preprocessing2.remove_diacritics]

    def lemmatize_text(text):
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        return ' '.join(word for word in [simplemma.lemmatize(w, lang='fr') for w in w_tokenizer.tokenize(text)])
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    Descr = pd.Series([titre + ", " + resume])
    Descr = hero.clean(Descr, clean_pipeline)
    Descr = Descr.apply(lemmatize_text)
    
    return Descr

def predict_qdrant(qdrant, encoder, text, collection, subjectsMaxCount):
    """
    Exécute une recherche vectorielle dans la collection Qdrant spécifiée.
    
    Utilise l'API Query (query_points) de qdrant-client.
    
    Args:
        qdrant: Client QdrantClient initialisé.
        encoder: Modèle SentenceTransformer pour vectoriser le texte.
        text (str): Texte nettoyé et lemmatisé à rechercher.
        collection (str): Nom de la collection vectorielle dans Qdrant.
        subjectsMaxCount (int): Nombre maximum de résultats retournés.
        
    Returns:
        list[dict]: Liste de dictionnaires contenant le score de similarité et le payload.
    """
    response = qdrant.query_points(
        collection_name=collection,
        query=encoder.encode(text).tolist(),
        limit=subjectsMaxCount
    )
    
    load_items = []
    print("pour:" + collection)
    for hit in response.points:
        load_items.append({'score': hit.score, 'label': hit.payload})
        print({'score': hit.score, 'label': hit.payload})

    return load_items

def extract_json_qdrant(text):
     
    global df_res
    load_items = []
    
    for items in text:
        load_items.append({'label': items['label']['a'],'score': items['score']  })

    z = pd.DataFrame(load_items)
  
    df_res = pd.concat([df_res, z], ignore_index=True)

    return z
    
def process_label(label):
    regexppn = '[0-9]{8}([0-9]|X){1}'
    mots_ids = label.split('--')  # Séparation en cas de plusieurs mots#id
    print("")
    print("mots_ids",mots_ids)
    mots = [mot_id.split('#')[0] for mot_id in mots_ids]  # Extraction des mots
    ids = [mot_id.split('#')[1] for mot_id in mots_ids]  # Extraction des IDs
    # swap of ids and mots list because somme models are built this way : 027229629#Bureaucratie, and not #Bureaucratie#02722962 !! TODO harmoniser
    if bool(re.fullmatch(regexppn, mots[0])) :         
        ids2switch = ids
        ids = mots
        mots = ids2switch
    return '--'.join(mots), '--'.join(ids)

def embedding_qdrant(titre, resume, qdrant, encoder, collection, subjectsMaxCount):

    
    # Vérifie si le résumé est présent
    if resume:
        DESCR = clean_and_lemmatize(titre, resume)
    else:
        DESCR = clean_and_lemmatize(titre, '')  # Si le résumé est absent, passe une chaîne vide
    '''
    if resume:
        DESCR = pd.Series([titre + ". " + resume])
    else:
        DESCR = pd.Series([titre + ". " + ""])
    '''
    
    valeur_specifique = DESCR[0]
    chaine_de_caracteres = str(valeur_specifique)
    data = {'result': [chaine_de_caracteres]}
    df = pd.DataFrame(data)

    global df_res
    df_res = pd.DataFrame()
    df['PREDICT'] = df['result'].apply(lambda x: predict_qdrant(qdrant, encoder, x, collection, subjectsMaxCount))

    df_test2 = df[['PREDICT']]
    print(df_test2)

    df_test2.apply(lambda x: extract_json_qdrant(x.PREDICT), axis=1)
    df_res['label'] = df_res['label'].replace('#u#', '_', regex=True).replace('#d#', '"', regex=True).replace(
        '#c#', '\'', regex=True).replace('_', ' ', regex=True)
    df_res['label'], df_res['id'] = zip(*df_res['label'].apply(process_label))
  #  df_res['id'] = df_res['label'].str.split('#').str[1]
   # df_res['label'] = df_res['label'].str.split('#').str[0]
  
    df_res = df_res[['label', 'id', 'score']]
    return df_res



