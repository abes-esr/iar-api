import os
import sys
import time
import ast
import json
import pickle
import datetime
import configparser
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from typing import List, Union
from time import sleep, perf_counter as pc

from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from langdetect import detect
from openai import OpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from .embed_lib import embedding_faiss, embedding_qdrant
except ImportError:
    from embed_lib import embedding_faiss, embedding_qdrant

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

# Chargement des index FAISS
conceptsORchains = "concepts"
avec_these = 'only_mono'

alias_model = 'distiluse'
vector_model = f"{conceptsORchains}_{alias_model}_{avec_these}"
index_distiluse = faiss.read_index(f"../rameau_vectorize_service/{vector_model}.faiss")
with open(f"../rameau_vectorize_service/{vector_model}_int2label.pkl", "rb") as f:
    int2label_distiluse = pickle.load(f)
label2int_distiluse = {v: k for k, v in int2label_distiluse.items()}

alias_model = 'allMin'
vector_model = f"{conceptsORchains}_{alias_model}_{avec_these}"
index_allMin = faiss.read_index(f"../rameau_vectorize_service/{vector_model}.faiss")
with open(f"../rameau_vectorize_service/{vector_model}_int2label.pkl", "rb") as f:
    int2label_allMin = pickle.load(f)
label2int_allMin = {v: k for k, v in int2label_allMin.items()}

alias_model = 'e5-large'
vector_model = f"{conceptsORchains}_{alias_model}_{avec_these}"
index_e5_large = faiss.read_index(f"../rameau_vectorize_service/{vector_model}.faiss")
with open(f"../rameau_vectorize_service/{vector_model}_int2label.pkl", "rb") as f:
    int2label_e5_large = pickle.load(f)
label2int_e5_large = {v: k for k, v in int2label_e5_large.items()}


def process_embedding_faiss(Title, Summary, client, encoder, collection, subjectsMaxCount):
    df_res = embedding_faiss(Title, Summary, client, encoder, collection, subjectsMaxCount)
    result = df_res.to_dict(orient="records")
    return result, df_res['label'].tolist()


def process_embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount):
    df_res = embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount)
    result = df_res.to_dict(orient="records")
    return result, df_res['label'].tolist()


def get_lang(text):
    try:
        return detect(text)
    except Exception:
        return ""


# Configuration du client LLM sécurisée via .env
ABES_LLM_URL = os.getenv("ABES_LLM_URL", "https://llm.ilaas.fr/v1")
ABES_LLM_KEY = os.getenv("ABES_LLM_KEY", "")
ABES_MODEL = os.getenv("ABES_LLM_MODEL", "llama-3.1-8b")

client = OpenAI(
    base_url=ABES_LLM_URL,
    api_key=ABES_LLM_KEY,
)


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


def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def embed_select(text, propositions, instruction):
    t0 = pc()
    all_texts = [text] + propositions
    embeddings = encoder3.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    title_embedding = embeddings[0]
    subject_embeddings = embeddings[1:]
    scores = [cosine_similarity(title_embedding, subj_emb) for subj_emb in subject_embeddings]
    reranked = sorted(zip(propositions, scores), key=lambda x: -x[1])

    resu = [label for label, score in reranked]
    print(f"Embed select time: {pc() - t0:.4f}s")
    return resu


def llm_select(text, propositions, instruction):
    t0 = pc()
    content = f"Voici le résumé d'un document : \n{text}\n---\n{instruction} :\n{propositions}\n---\nRéponds de manière brève, en ne mettant que la liste des mots-clés qui ne représentent pas le contenu du résumé."
    resu = "error acces llm"
    try:
        response = client.chat.completions.create(
            model=ABES_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that is designed to output simple lists, with items separated by comma. Do not output anything else."},
                {"role": "user", "content": content}
            ]
        )
        resu = response.choices[0].message.content
    except Exception as e:
        print(f"Erreur appel LLM: {e}")
    print(f"LLM time: {pc() - t0:.4f}s")
    return resu


def llm_postprocess(llm_resu, union_list):
    try:
        union_labels = [x['label'] for x in union_list]
        intrus = llm_resu.split(",")
        intrus_clean = [kw.strip() for kw in intrus]
        union_labels_postLLM = list(set(union_labels) - set(intrus_clean))
        return [idlabel for idlabel in union_list if idlabel['label'] in union_labels_postLLM]
    except Exception as e:
        print(e)
        return union_list


def cross_postprocess(llm_resu, union_list):
    try:
        ordre = {valeur: index for index, valeur in enumerate(llm_resu)}
        return sorted(union_list, key=lambda x: ordre[x['label']])
    except Exception as e:
        print(e)
        return union_list


def emb_postprocess(llm_resu, union_list):
    try:
        ordre = {valeur: index for index, valeur in enumerate(llm_resu)}
        return sorted(union_list, key=lambda x: ordre[x['label']])
    except Exception as e:
        print(e)
        return union_list


def dollar3format(ppn, format):
    if format == "html":
        return f'<a href="http://www.idref.fr/{ppn}">{ppn}</a>'
    return ppn


def set606(id, label, formatt):
    z606 = ""
    i = 0
    if "--" in id:
        z606 = "606 ##"
        ppns_list = id.split("--")
        labels_list = label.split("--")
        for x in ppns_list:
            z606 += f"$3{dollar3format(ppns_list[i], formatt)}$a{labels_list[i]}"
            i += 1
        z606 += "$2rameau\n"
    else:
        if id in subdivisionsONLY_list:
            z606 = f"$3{dollar3format(id, formatt)}$x{label}"
        else:
            z606 = f"606 ##$3{dollar3format(id, formatt)}$a{label}"
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


def display_roots(root, not_displayed, pred_dict, formatt):
    try:
        not_displayed.remove(root)
    except Exception:
        pass

    display_root = set606(root, pred_dict[root], formatt)
    display_root_children = [display_root]
    getChildren(root, 0, not_displayed, 0, pred_dict, display_root_children, formatt)
    return display_root_children


def getChildren(parent, level, not_displayed, indent, pred_dict, display_root_children, formatt):
    retrait = '&nbsp;&nbsp;&nbsp;' if formatt == 'html' else '   '

    if parent in rameau_parents_grouped.index:
        children = rameau_parents_grouped.loc[parent, 'PPN']
        level += 1
        indent += 1
        for child in children:
            if child in pred_dict.keys():
                if child in not_displayed:
                    display_root_children.append(retrait + set606(child, pred_dict[child], formatt))
                    try:
                        not_displayed.remove(child)
                    except Exception:
                        pass
                    if level < 4:
                        getChildren(child, level, not_displayed, indent, pred_dict, display_root_children, formatt)
            else:
                if level < 4:
                    getChildren(child, level, not_displayed, 0, pred_dict, display_root_children, formatt)


def other_suggestions(predictions, top_ppns):
    bloc_model = ""
    if len(predictions) > 0:
        for prediction in predictions:
            ppns = prediction["id"]
            labels = prediction["label"]
            if ppns not in top_ppns:
                bloc_model += set606(ppns, labels)
    return bloc_model


def jsonTo606(data, structure, formatt):
    pred_dict = {}

    keys_to_check = ["union", "intersection", "intersection2models", "intersection2models1best", "cross", "emb", "llm"]
    for key in keys_to_check:
        if key in data["PredictionByAggregation"] and len(data["PredictionByAggregation"][key]) > 0:
            pred_dict = {prediction['id']: prediction['label'] for prediction in data["PredictionByAggregation"][key]}

    suggested_ids = list(pred_dict.keys())

    if structure == 'flat':
        return [set606(s_id, pred_dict[s_id], formatt) for s_id in suggested_ids]

    elif structure == 'tree':
        not_displayed = suggested_ids.copy()
        suggested_roots = getRoots(pred_dict)
        display = []

        for root_id in suggested_roots:
            display_root_children = display_roots(root_id, not_displayed, pred_dict, formatt)
            display.extend(display_root_children)

        for not_displayed_id in not_displayed:
            display.append(set606(not_displayed_id, pred_dict[not_displayed_id], formatt))

        return display


instruction = "Parmi la liste de mots-clés qui suit, donne-moi la sous-liste des mots-clés qui ne représentent pas le contenu du résumé"


@app.get("/subject_indexation/")
async def index_subjects(
    Title: str,
    Summary: str = None,
    docId: str = '',
    models: str = None,
    aggregationType: str = None,
    vocabulary: str = 'rameau',
    subjectsMaxCount: int = 5,
    Agent: str = '',
    Format: str = 'json',
    request: Request = None
):
    liste_models = ['victor3_chain', 'victor1_concept', 'victor1_chain', 'victor2']
    liste_agregation = ['union', 'intersection', 'intersection2models', 'intersection2models1best', 'llm', 'lilimarlene', 'cross', 'qdrant', 'faiss', 'embbed']
    liste_models_best = ['victor1_concept', 'victor3_chain']
    liste_complementarymodels = ['victor2']

    result_data = {}
    response_data = {
        "DocumentID": docId,
        "PredictionByModel": {},
        "PredictionByAggregation": {}
    }

    vocabulary = vocabulary.lower()

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

    utiliser_faiss = 'qdrant' not in aggregationType if aggregationType else True

    if aggregationType:
        for aggregation in types_propres:
            if aggregation.strip() not in liste_agregation:
                raise HTTPException(status_code=404, detail=f"L'agrégation '{aggregation.strip()}' n'est pas disponible.")

    if vocabulary == 'rameau':
        for model_name in liste_models_in_param:
            start_time_model = time.time()
            if model_name in ['victor1_concept', 'victor1_chain']:
                if utiliser_faiss:
                    result_data[model_name], _ = process_embedding_faiss(Title, Summary, index_allMin, encoder1, int2label_allMin, subjectsMaxCount)
                else:
                    result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder1, 'concepts_allMin_only_mono', subjectsMaxCount)
            elif model_name == 'victor2':
                if utiliser_faiss:
                    result_data[model_name], _ = process_embedding_faiss(Title, Summary, index_distiluse, encoder2, int2label_distiluse, subjectsMaxCount)
                else:
                    result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder2, 'concepts_distiluse_only_mono', subjectsMaxCount)
            elif model_name in ['victor3_chain', 'victor3_chain_en']:
                if utiliser_faiss:
                    result_data[model_name], _ = process_embedding_faiss(Title, Summary, index_e5_large, encoder3, int2label_e5_large, subjectsMaxCount)
                else:
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

    model_results_best = [result_data[m] for m in result_data if m in liste_models_best]
    union_labels_ids_best = {(item['label'], item['id']) for sublist in model_results_best for item in sublist}
    union_list_best = [{"label": label, "id": id} for label, id in union_labels_ids_best]

    intersection_labels_ids = set((item['label'], item['id']) for item in model_results[0]) if model_results else set()
    for sublist in model_results[1:]:
        intersection_labels_ids &= set((item['label'], item['id']) for item in sublist)
    intersection_list = [{"label": label, "id": id} for label, id in intersection_labels_ids]

    resultats_2models = set()
    for model_name, result in result_data.items():
        for item in result:
            element, element_id = item['label'], item['id']
            if any(element == autre_item['label'] for autre_m, autre_res in result_data.items() if autre_m != model_name for autre_item in autre_res):
                resultats_2models.add((element, element_id))
    resultats_2models_list = [{"label": label, "id": id} for label, id in resultats_2models]

    resultats_best = set()
    for model_name, result in result_data.items():
        if model_name in liste_models_best:
            for item in result:
                element, element_id = item['label'], item['id']
                if any(element == autre_item['label'] for autre_m, autre_res in result_data.items() if autre_m in liste_complementarymodels for autre_item in autre_res):
                    resultats_best.add((element, element_id))
    resultats_best_list = [{"label": label, "id": id} for label, id in resultats_best]

    list_a_traiter = []
    if aggregationType:
        aggregated_predictions = {}

        if "union" in types_propres:
            aggregated_predictions["union"] = union_list
            list_a_traiter = union_list

        if "intersection" in types_propres:
            aggregated_predictions["intersection"] = intersection_list
            list_a_traiter = intersection_list

        if "intersection2models" in types_propres:
            aggregated_predictions["intersection2models"] = resultats_2models_list
            list_a_traiter = resultats_2models_list

        if "intersection2models1best" in types_propres:
            aggregated_predictions["intersection2Models1Best"] = resultats_best_list
            list_a_traiter = resultats_best_list

        labels_a_traiter = [x['label'] for x in list_a_traiter]

        if "llm" in types_propres or "lilimarlene" in types_propres:
            aggregated_predictions["llm"] = llm_postprocess(llm_select(f"{Title}. {Summary}", ", ".join(labels_a_traiter), instruction), list_a_traiter)

        if "cross" in types_propres:
            tmp = cross_select(f"{Title}. {Summary}", labels_a_traiter, list_a_traiter)
            aggregated_predictions["cross"] = cross_postprocess(tmp, list_a_traiter)

        if "embbed" in types_propres:
            tmp = embed_select(f"{Title}. {Summary}", labels_a_traiter, list_a_traiter)
            aggregated_predictions["emb"] = emb_postprocess(tmp, list_a_traiter)

        response_data["PredictionByAggregation"] = aggregated_predictions

    # Écriture dans le fichier d'historique
    instant = datetime.datetime.now()
    file_path = root + "history.txt"
    with open(file_path, 'a') as file:
        try:
            file.write(f"\n{Agent};{docId};{Title.replace(';', ',')};{Summary.replace(';', ',')};{instant}")
        except Exception as e:
            print(f"Erreur écriture historique: {e}")

    # Réponse sous différents formats
    if Format == "text":
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'tree', 'text')), status_code=200)
    if Format == "text_flat":
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'flat', 'text')), status_code=200)
    if Format == "text_tree":
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'tree', 'text')), status_code=200)
    if Format in ["html", "html_flat", "html_tree"]:
        struct = 'flat' if Format == "html_flat" else 'tree'
        page = f"<!DOCTYPE html><html><head><title>Propositions</title><meta charset='utf-8'/></head><body>{''.join(jsonTo606(response_data, struct, 'html')).replace(chr(10), '<br/>')}</body></html>"
        return HTMLResponse(content=page, status_code=200)

    return JSONResponse(content=response_data)


rameau_html = """
<!DOCTYPE html>
<html>
<body>
<h2>Rameau</h2>
<div id="rm">
<p>RCR</p><textarea id="rcr" name="rcr" rows="1" cols="20"></textarea>
<p>Titre</p><textarea id="titre" name="titre" rows="3" cols="200"></textarea>
<p>Résumé</p><textarea id="resume" name="resume" rows="3" cols="200"></textarea>
<p>Nombre de résultats par modèle</p><textarea id="hitscount" name="hitscount" rows="1" cols="20">5</textarea>
<br/>
<button type="button" onclick="loadRameau()">envoi</button>
<p>Rameau</p>
<div id="resu"></div>
</div>
<script>
function loadRameau() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    document.getElementById("resu").innerHTML = this.responseText;
  }
  xhttp.open("GET","/subject_indexation/?docId=null&Title="+encodeURIComponent(document.getElementById("titre").value)+"&Summary="+encodeURIComponent(document.getElementById("resume").value)+"&models=victor3_chain,victor1_concept,victor2&aggregationType=intersection2models,llm&subjectsMaxCount="+encodeURIComponent(document.getElementById("hitscount").value)+"&Agent="+document.getElementById("rcr").value+"&vocabulary=rameau&Format=html");
  xhttp.send();
}
</script>
</body>
</html>
"""


@app.get("/rameau")
async def present_form_rameau():
    return HTMLResponse(rameau_html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8071)