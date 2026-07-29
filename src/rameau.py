import os
import time
import ast
import pickle
import datetime
import configparser
import pandas as pd
from pathlib import Path
from time import perf_counter as pc

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from langdetect import detect
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from .embed_lib import embedding_qdrant
except ImportError:
    from embed_lib import embedding_qdrant

#avec cross et faiss
# http://labo-vm-1.v202.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=union,cross,faiss&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree

#avec llm et gdrant
# http://labo-vm-1.v202.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=intersection2models,llm,qdrant&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree

# Chargement des variables d'environnement (.env)
load_dotenv()

app = FastAPI(title="API Indexation Sujet RAMEAU")

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root = ''
root1 = ''
utiliser_faiss = False

cwd = str(Path.cwd())
print("cwd", cwd)
if "/app" in cwd:
    root = '/app/'
    root1 = '/app/'

# Configuration Qdrant
config_init = configparser.ConfigParser()
adress_qdrant = os.getenv("QDRANT_HOST", "localhost")
port_qdrant = int(os.getenv("QDRANT_PORT", 6333))

try:
    config_init.read('global_init.ini')
except Exception:
    config_init['DEFAULT'] = {'adress_qdrant': adress_qdrant}
    with open(root + 'global_init.ini', 'w') as configfile:
        config_init.write(configfile)

Qdrant_Client = QdrantClient(host=adress_qdrant, port=port_qdrant)

# Chargement des modèles SentenceTransformers / CrossEncoder
encoder1 = SentenceTransformer('all-MiniLM-L6-v2')
encoder2 = SentenceTransformer('distiluse-base-multilingual-cased-v2')
encoder3 = SentenceTransformer('intfloat/multilingual-e5-large')
encoder4 = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Importation des vectoriseurs TF-IDF et encodeurs de labels
with open(root + 'vectorizer_ppn.pkl', 'rb') as f:
    vectorizer1 = pickle.load(f)

with open(root + 'label_encoder_ppn.pkl', 'rb') as f:
    le1 = pickle.load(f)

with open(root + 'vectorizer_ppn_100.pkl', 'rb') as f:
    vectorizer2 = pickle.load(f)

with open(root + 'label_encoder_ppn_100.pkl', 'rb') as f:
    le2 = pickle.load(f)

# Dictionnaires et données RAMEAU
rameau_ancestors_df = pd.read_csv(root + "rameau_ancestors_df.csv")
rameau_ancestors_df.set_index("ppn", inplace=True)

rameau_parents = pd.read_csv(root + "rameau_parents.csv")
rameau_parents_grouped = rameau_parents.groupby('PPN_LINKED')['PPN'].apply(list).reset_index()
rameau_parents_grouped.set_index("PPN_LINKED", inplace=True)

subdivisionsONLY_df = pd.read_csv(root + "rameau_subdivisionsONLY.csv")
subdivisionsONLY_list = list(subdivisionsONLY_df['PPN'])




def process_embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount):
    df_res = embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount)
    result = df_res.to_dict(orient="records")
    return result, df_res['label'].tolist()


def get_lang(text):
    try:
        return detect(text)
    except Exception:
        return ""





def cross_select(text, propositions, instruction):
    t0 = pc()
    pairs = [[text, label] for label in propositions]
    scores = encoder4.predict(pairs)
    reranked = sorted(zip(propositions, scores), key=lambda x: -x[1])

    resu = []
    for label, score in reranked:
        resu.append(label)

    print(f"Cross-encoder execution time: {pc() - t0:.4f}s")
    return resu




def cross_postprocess(llm_resu, union_list):
    try:
        ordre = {valeur: index for index, valeur in enumerate(llm_resu)}
        return sorted(union_list, key=lambda x: ordre[x['label']])
    except Exception as e:
        print(e)
        return union_list



def set606(id, label):
    z606 = ""
    i = 0
    if "--" in id:
        z606 = "606 ##"
        ppns_list = id.split("--")
        labels_list = label.split("--")
        for x in ppns_list:
            z606 += f"$3{ppns_list[i]}$a{labels_list[i]}"
            i += 1
        z606 += "$2rameau\n"
    else:
        if id in subdivisionsONLY_list:
            z606 = f"$3{id}$x{label}"
        else:
            z606 = f"606 ##$3{id}$a{label}"
        z606 += "$2rameau\n"
    return z606


def getRoots(pred_dict):
    suggested_roots = []
    suggested_ids = list(pred_dict.keys())

    for suggested_id in suggested_ids:
        if '--' in suggested_id:
            suggested_roots.append(suggested_id)
        else:
            try:
                ancestors = rameau_ancestors_df.loc[suggested_id]['ancestors']
                ancestors_list = ast.literal_eval(ancestors)
                ancestors_among_suggested = [a for a in ancestors_list if a in suggested_ids and a != suggested_id]
                if not ancestors_among_suggested:
                    suggested_roots.append(suggested_id)
            except Exception:
                suggested_roots.append(suggested_id)

    return suggested_roots


def display_roots(root, not_displayed, pred_dict):
    try:
        not_displayed.remove(root)
    except Exception:
        pass

    display_root = set606(root, pred_dict[root])
    display_root_children = [display_root]
    getChildren(root, 0, not_displayed, 0, pred_dict, display_root_children)
    return display_root_children


def getChildren(parent, level, not_displayed, indent, pred_dict, display_root_children):
    retrait = '   '

    if parent in rameau_parents_grouped.index:
        children = rameau_parents_grouped.loc[parent, 'PPN']
        level += 1
        indent += 1
        for child in children:
            if child in pred_dict.keys():
                if child in not_displayed:
                    display_root_children.append(retrait + set606(child, pred_dict[child]))
                    try:
                        not_displayed.remove(child)
                    except Exception:
                        pass
                    if level < 4:
                        getChildren(child, level, not_displayed, indent, pred_dict, display_root_children)
            else:
                if level < 4:
                    getChildren(child, level, not_displayed, 0, pred_dict, display_root_children)


def jsonTo606(data, structure):
    pred_dict = {}

    keys_to_check = ["union", "cross"]
    for key in keys_to_check:
        if key in data["PredictionByAggregation"] and len(data["PredictionByAggregation"][key]) > 0:
            pred_dict = {prediction['id']: prediction['label'] for prediction in data["PredictionByAggregation"][key]}

    suggested_ids = list(pred_dict.keys())

    if structure == 'flat':
        return [set606(s_id, pred_dict[s_id]) for s_id in suggested_ids]

    elif structure == 'tree':
        not_displayed = suggested_ids.copy()
        suggested_roots = getRoots(pred_dict)
        display = []

        for root_id in suggested_roots:
            display_root_children = display_roots(root_id, not_displayed, pred_dict)
            display.extend(display_root_children)

        for not_displayed_id in not_displayed:
            display.append(set606(not_displayed_id, pred_dict[not_displayed_id]))

        return display


instruction = "Parmi la liste de mots-clés qui suit, donne-moi la sous-liste des mots-clés qui ne représentent pas le contenu du résumé"


@app.get("/subject_indexation/")
# ==============================================================================
# ROUTE PRINCIPALE D'INDEXATION RAMEAU
# ==============================================================================

@app.get("/subject_indexation/")
async def index_subjects(
    Title: str,
    Summary: str = None,
    docId: str = '',
    models: str = None,
    aggregationType: str = None,
    subjectsMaxCount: int = 5,
    Agent: str = '',
    Format: str = 'json',
    request: Request = None
):
    """
    Endpoint principal pour l'indexation de sujets RAMEAU.
    
    Cette fonction reçoit le titre, le résumé, sélectionne les modèles vectoriels
    pertinents, et effectue des calculs de similarité à l'aide de Qdrant.
    Elle effectue également des agrégations si demandé (ex: union, cross)
    et formate le résultat (JSON ou format textuel UNIMARC 606).
    """
    liste_models = ['victor3_chain', 'victor1_concept', 'victor1_chain', 'victor2']
    liste_agregation = ['union', 'cross', 'qdrant']
    liste_models_best = ['victor1_concept', 'victor3_chain']

    result_data = {}
    response_data = {
        "DocumentID": docId,
        "PredictionByModel": {},
        "PredictionByAggregation": {}
    }

    if models is None:
        liste_models_in_param = [x for x in liste_models_best]
    elif models == "*":
        liste_models_in_param = [x for x in liste_models]
    else:
        liste_models_in_param = [m.strip().lower() for m in models.split(',')]

    if Summary is None:
        Summary = ""

    if get_lang(Title) != "fr" or (Summary != "" and get_lang(Summary) != "fr"):
        liste_models_in_param.insert(0, "victor3_chain_en")
        liste_models.insert(0, "victor3_chain_en")
        liste_models_best.insert(0, "victor3_chain_en")

    if liste_models_in_param:
        for model in liste_models_in_param:
            if model.strip() not in liste_models:
                raise HTTPException(status_code=404, detail=f"Le modèle '{model.strip()}' n'est pas disponible.")

    RCRs = ["991279901", "341720001", "RCR", "040702201", "340322101", "350472301", "751050006", "991270001"]

    if Agent == "":
        return HTMLResponse(content="Init avec agent null ok.", status_code=200)

    if Agent not in RCRs:
        return HTMLResponse(content="Votre bibliothèque ne peut accéder à ce service réservé aux établissements testeurs.", status_code=200)

    if aggregationType is not None:
        cleaned_types = aggregationType.lower()
        types_list = liste_agregation if aggregationType == "*" else cleaned_types.split(',')
        types_propres = [element.strip() for element in types_list]
    else:
        types_propres = []

    if aggregationType:
        for aggregation in types_propres:
            if aggregation.strip() not in liste_agregation:
                raise HTTPException(status_code=404, detail=f"L'agrégation '{aggregation.strip()}' n'est pas disponible.")

    # Interrogation des modèles vectoriels uniquement via Qdrant
    for model_name in liste_models_in_param:
        start_time_model = time.time()
        if model_name in ['victor1_concept', 'victor1_chain']:
            result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder1, 'concepts_allMin_only_mono', subjectsMaxCount)
        elif model_name == 'victor2':
            result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder2, 'concepts_distiluse_only_mono', subjectsMaxCount)
        elif model_name in ['victor3_chain', 'victor3_chain_en']:
            result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder3, 'concepts_e5-large_only_mono', subjectsMaxCount)
        end_time_model = time.time()

        if subjectsMaxCount is not None and subjectsMaxCount > 0:
            result_data[model_name] = result_data[model_name][:subjectsMaxCount]

        response_data["PredictionByModel"][model_name] = {
            "Result": result_data[model_name],
            "ResponseTime": f"{round(end_time_model - start_time_model, 2)} secondes"
        }

    model_results = list(result_data.values())
    union_labels_ids = {(item['label'], item['id']) for sublist in model_results for item in sublist}
    union_list = [{"label": label, "id": id} for label, id in union_labels_ids]

    list_a_traiter = []
    if aggregationType:
        aggregated_predictions = {}

        if "union" in types_propres:
            aggregated_predictions["union"] = union_list
            list_a_traiter = union_list

        labels_a_traiter = [x['label'] for x in list_a_traiter]

        if "cross" in types_propres:
            tmp = cross_select(f"{Title}. {Summary}", labels_a_traiter, list_a_traiter)
            aggregated_predictions["cross"] = cross_postprocess(tmp, list_a_traiter)

        response_data["PredictionByAggregation"] = aggregated_predictions

    # Écriture dans le fichier d'historique
    instant = datetime.datetime.now()
    file_path = root + "history.txt"
    with open(file_path, 'a') as file:
        try:
            file.write(f"\n{Agent};{docId};{Title.replace(';', ',')};{Summary.replace(';', ',')};{instant}")
        except Exception as e:
            print(f"Erreur écriture historique: {e}")

    # Réponse sous différents formats (textuels uniquement)
    if Format == "text":
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'tree')), status_code=200)

    return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8071)