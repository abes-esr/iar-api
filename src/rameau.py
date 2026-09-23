from dotenv import load_dotenv
import os
import sys
from pathlib import Path

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json           
from embed_lib import *
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
from time import perf_counter as pc
import datetime                       
from langdetect import detect
from openai import OpenAI
import ast


#avec cross et faiss
# http://labo-vm-1.v202.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=union,cross,faiss&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree

#avec llm et gdrant
# http://labo-vm-1.v202.abes.fr:8071/subject_indexation/?docId=NULL&Title=Alg%C3%A8bre%20vectorielle&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=intersection2models,llm,qdrant&subjects&MaxCount=10&Agent=RCR&vocabulary=rameau&Format=text_tree


app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adress_qdrant = os.getenv("IAR_QDRANT_HOST", os.getenv("QDRANT_HOST", "localhost"))
port_qdrant = int(os.getenv("IAR_QDRANT_PORT", os.getenv("QDRANT_PORT", 6333)))
                
Qdrant_Client = QdrantClient(host=adress_qdrant, port=port_qdrant)

# Initialisation des modèles d'embedding et de reranking
encoder1 = SentenceTransformer('all-MiniLM-L6-v2')
encoder2 = SentenceTransformer('distiluse-base-multilingual-cased-v2')
encoder3 = SentenceTransformer('intfloat/multilingual-e5-large')
encoder4 = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")


def get_csv_dir() -> Path:
    """
    Localise le répertoire contenant les fichiers CSV de référence RAMEAU.
    
    Vérifie en priorité la variable d'environnement DATA_DIR, puis les chemins standards :
    - Dans le conteneur Docker : /app/data/csv (issu du montage volume), puis /app/data
    - En local : volumes/csv ou volumes à la racine du projet
    
    Returns:
        Path: Le chemin résolu vers le répertoire contenant les fichiers CSV.
    """
    env_data_dir = os.getenv("DATA_DIR", "").strip()
    candidates = []
    
    # 1. Vérification si DATA_DIR est défini (hors valeur par défaut '.')
    if env_data_dir and env_data_dir != ".":
        candidates.extend([Path(env_data_dir) / "csv", Path(env_data_dir)])
        
    # 2. Chemins standards selon l'environnement
    if "/app" in str(Path.cwd()):
        candidates.extend([Path("/app/data/csv"), Path("/app/data"), Path("/app")])
    else:
        base_dir = Path(__file__).resolve().parent.parent
        candidates.extend([
            base_dir / "volumes" / "csv",
            base_dir / "volumes",
            Path("volumes/csv"),
            Path("volumes"),
            Path(".")
        ])
        
    for candidate in candidates:
        if (candidate / "rameau_ancestors_df.csv").exists():
            return candidate
            
    # Chemin par défaut si non trouvé
    return Path("/app/data/csv") if "/app" in str(Path.cwd()) else Path("volumes/csv")


csv_dir = get_csv_dir()
print(f"[RAMEAU] Répertoire CSV utilisé : {csv_dir.resolve()}")

# Dictionnaire des ancêtres de chaque Rameau
rameau_ancestors_df = pd.read_csv(csv_dir / "rameau_ancestors_df.csv")
rameau_ancestors_df.set_index("ppn", inplace=True)

# Tableau à deux colonnes : un concept et son parent (un concept peut avoir plusieurs parents)
rameau_parents = pd.read_csv(csv_dir / "rameau_parents.csv")
rameau_parents_grouped = rameau_parents.groupby('PPN_LINKED')['PPN'].apply(list)
rameau_parents_grouped = rameau_parents_grouped.reset_index()
rameau_parents_grouped.set_index("PPN_LINKED", inplace=True)

# Liste des concepts Rameau qu'on ne peut employer que comme subdivision
subdivisionsONLY_df = pd.read_csv(csv_dir / "rameau_subdivisionsONLY.csv")
subdivisionsONLY_list = list(subdivisionsONLY_df['PPN'])
    
#Prediction en utilsant les embedding
def process_embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount):
    df_res = embedding_qdrant(Title, Summary, qdrant_client, encoder, collection, subjectsMaxCount)
    result = df_res.to_dict(orient="records")
    return result, df_res['label'].tolist()

# Test de la langue
def get_lang(text) :
    try :
        lang = detect(text)
        textlang = lang
    except  :
        textlang = ""
    return textlang
print("zise of get_lang")
print(sys.getsizeof(get_lang))

#### LLM ###

abes_llama3 = os.getenv('IAR_LLM_URL', os.getenv('ABES_LLM_URL', 'http://iar-llm:11434/v1'))
key_abes_llama3 = os.getenv('IAR_LLM_KEY', os.getenv('ABES_LLM_KEY', 'ollama'))
abes_model = os.getenv('IAR_LLM_MODEL', os.getenv('ABES_LLM_MODEL', 'llama-3.1-8b'))

# Créer une connexion à un LLM:
client = OpenAI(
    base_url = abes_llama3,
    api_key= key_abes_llama3, # required, but unused
)

def cross_select(text, propositions, instruction):
    
    t0 = pc() # mesurer le temps de traitement
    # Construire les couples (document, label)
    pairs = [[text, label] for label in propositions]
    print("pairs",pairs)
    # Scores du cross-encoder
    scores = encoder4.predict(pairs)
    # Coupler scores + labels
    reranked = sorted(zip(propositions, scores), key=lambda x: -x[1])

    print("Classement final :")
    resu = []
    for label, score in reranked:
        print(f"{label:30s}  score={score:.4f}")
        resu.append(label)
    
    print(pc()-t0)
    return resu

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def embed_select(text, propositions, instruction):
    t0 = pc() # mesurer le temps de traitement
    all_texts = [text] + propositions
    print("pairs",all_texts)
    embeddings = encoder3.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True)
    title_embedding = embeddings[0]
    subject_embeddings = embeddings[1:]
    scores = [cosine_similarity(title_embedding, subj_emb) for subj_emb in subject_embeddings]
    reranked = sorted(zip(propositions, scores), key=lambda x: -x[1])
    print("Classement final :")
    resu = []
    for label, score in reranked:
        print(f"{label:30s}  score={score:.4f}")
        resu.append(label)
    
    print(pc()-t0)
    return resu
    
    
def llm_select(text, propositions, instruction):
    t0 = pc() # mesurer le temps de traitement
    content = "Voici le résumé d'un document : \n"+text+"\n---\n"+instruction+" :\n"+propositions+"\n---\n"+"Réponds de manière brève, en ne mettant que la liste des mots-clés qui ne représentent pas le contenu du résumé."
    #print(content)
    resu="error acces llm"
    try :
        response = client.chat.completions.create(
          model= abes_model,
         temperature = 0.1,
          messages=[
            {"role": "system", "content": "You are a helpful assistant that is designed to output simple lists, with items separated by comma. Do not output anything else."},
            {"role": "user", "content": content}
          ]
        )
        resu = response.choices[0].message.content
    except Exception as e :
        print(e)
    print(pc()-t0)
    return resu

# On passe union_list_best et non union_list (sauf exception)
def llm_postprocess(llm_resu, union_list):
    print("llm_resu : **" + str(llm_resu) + "**")
    #print("union_list : "+str(union_list))
    try :
        union_labels = [x['label'] for x in union_list]
        print(". union_labels : "+str(union_labels))
        intrus = llm_resu.split(",")
        intrus_clean = [kw.strip() for kw in intrus]
        print(". intrus_clean : "+str(intrus_clean))
        union_labels_postLLM = list(set(union_labels) - set(intrus_clean))
        print(".. union_labels_postLLM : "+str(union_labels_postLLM))
        union_list_postLLM = []
        for idlabel in union_list :
            #print(idlabel)
            if idlabel['label'] in union_labels_postLLM :
                union_list_postLLM.append(idlabel)
        print("... union_list_postLLM : "+str(union_list_postLLM))
        return union_list_postLLM
        #union_labels = [x['label'] for x in union_list]
    except Exception as e :
        print(e)
        
def cross_postprocess(llm_resu, union_list):
    print("cross_resu : **" ,llm_resu)
    print("union_list : ",union_list)
    try :
        print("trie...")
        ordre = {valeur: index for index, valeur in enumerate(llm_resu)}
        #trie = sorted(union_list, key=lambda x: llm_resu.index(union_list['label']))
        trie = sorted(union_list, key=lambda x: ordre[x['label']])
        print("trie",trie)
        return trie
        
    except Exception as e :
        print(e)
def emb_postprocess(llm_resu, union_list):
    print("emb_resu : **" ,llm_resu)
    print("union_list : ",union_list)
    try :
        print("trie...")
        ordre = {valeur: index for index, valeur in enumerate(llm_resu)}
        #trie = sorted(union_list, key=lambda x: llm_resu.index(union_list['label']))
        trie = sorted(union_list, key=lambda x: ordre[x['label']])
        print("trie",trie)
        return trie
        
    except Exception as e :
        print(e)

def dollar3format(ppn, format) :
    if format == "html" :
        ppn2 = '<a href="http://www.idref.fr/'+ppn+'">'+ppn+'</a>'
    else : 
        ppn2 = ppn
    return ppn2

# Affichage de chaque indexation (une ligne)
#   . Prise en compte des concepts complexes : à découper
#   . Prise en compte des concepts qu'on ne peut employer que comme subdivision : pas de tag de zone, $x
#   . Prise en compte du formattage : texte ou html
def set606(id, label, formatt):
    #print(label)
    #print(id)
    z606 = ""
    
    i = 0
    if "--" in id:
        z606 = "606 ##"
        ppns_list = id.split("--")
        labels_list = label.split("--")
        for x in ppns_list:
            z606 = z606 + "$3"+ dollar3format(ppns_list[i], formatt) + "$a" + labels_list[i]
            i += 1
        z606 = z606 + "$2rameau\n"
    else :
        if id in subdivisionsONLY_list :
            z606 = "$3"+ dollar3format(id, formatt) + "$x" + label      # subvision only
        else :
            z606 = "606 ##$3"+ dollar3format(id, formatt) + "$a" + label
        i += 1
        z606 = z606 + "$2rameau\n"       
    return z606

# dans une liste de suggestions, identifier celles qui n'ont pas d'ancêtre dans les suggestions : les racines (roots)
def getRoots(pred_dict) :
    suggested_roots = []
    suggested_ids = pred_dict.keys()
    for suggested_id in list(suggested_ids) :
        print('suggested_id : ' + suggested_id + ' ' + pred_dict[suggested_id])
        #print(suggested_id)
        # les suggestions construites (avec --) sont considérées comme des roots
        if '--' in suggested_id :
            suggested_roots.append(suggested_id)
            #print('... is considered as a root')
        else :
            try :
                #ancestors = suggested_ancestors[suggested_ancestors['ppn'] == suggested_id]['ancestors'].values[0]
                #ancestors = rameau_ancestors_df[rameau_ancestors_df['ppn'] == suggested_id]['ancestors'].values[0]
                ancestors = rameau_ancestors_df.loc[suggested_id]['ancestors']
                ancestors_list = ast.literal_eval(ancestors)
                print(ancestors_list)
                ancestors_among_suggested = []
                for ancestor in ancestors_list :
                    #print(ancestor)
                    if ancestor in suggested_ids and ancestor != suggested_id :
                        ancestors_among_suggested.append(ancestor)
                        print('... has ancestor : ' + ancestor + ' ' + pred_dict[ancestor])
                    else :
                        pass
                if len(ancestors_among_suggested) == 0 :
                    suggested_roots.append(suggested_id)
                    print('... is a root')
            except :
                suggested_roots.append(suggested_id)
                print('... is considered as a root (because not found in suggested_ancestors, à creuser pourquoi')
    return suggested_roots

# Affichage des 606 en partant des concepts racine et en descendant, de manière récursive (jusu'à une certaine limite : 4 niveaux ?)
def display_roots(root, not_displayed, pred_dict, formatt) :
    #print(not_displayed)
    #print(root)
    #print(pred_dict[parent] + '('+parent+')')
    ''''''
    try :
        not_displayed.remove(root)
    except :
        pass
    #print(set606(root, pred_dict[root]))
    display_root = set606(root, pred_dict[root], formatt)
    display_root_children  = [display_root]
    #display_root_children.append('pomm')
    #print('----- DISPLAY_ROOT :')
    #print(display_root)
    #print('----- / DISPLAY_ROOT')
    getChildren(root, 0, not_displayed, 0, pred_dict, display_root_children, formatt)
    #print('----- DISPLAY_ROOT children :')
    #print(display_root_children)
    #print('----- / DISPLAY_ROOT children')'''
    return display_root_children

def getChildren(parent, level, not_displayed, indent, pred_dict, display_root_children, formatt) :
    #print('Les enfants de : '+parent)
    #print('display_root_children : '+str(display_root_children))

    # whitespace processing to format the hierarchy of suggestions
    if formatt == 'html' :
        retrait = '&nbsp;&nbsp;&nbsp;'   # space entity for html display
    else : 
        retrait = '   '
    
    if parent in rameau_parents_grouped.index :
        children = rameau_parents_grouped.loc[parent, 'PPN']
        #print("children :")
        #print(children)
        #print(type(children))
        level = level + 1
        indent = indent + 1
        for child in children :
            #print(str('._'*level) + child )
            if child in list(pred_dict.keys()) :
                print(child + ' est dans les suggestions')
                if child in not_displayed :
                    #print(str('.'*level) + pred_dict[child] + '('+child+')')
                    #print(str('.'*indent) + ' (enfant de '+parent + ') ' + 'level'+str(level) + ' ' + set606(child, pred_dict[child]) )
                    #display_root_children = display_root_children + set606(child, pred_dict[child])
                    #display_root_children.append(str('.'*indent) + ' (enfant de '+parent + ') ' + 'level'+str(level) + ' ' + set606(child, pred_dict[child], formatt))
                    display_root_children.append(retrait + set606(child, pred_dict[child], formatt))
                    #print('display_root_children : '+str(display_root_children))
                    ''''''
                    try :
                        not_displayed.remove(child)
                    except :
                        pass
                    
                    if level < 4 :
                        #level = level+1
                        getChildren(child, level, not_displayed, indent, pred_dict, display_root_children, formatt)
            else :
                if level < 4 :
                    getChildren(child, level, not_displayed, 0, pred_dict, display_root_children, formatt)   # 
        '''print('----- DISPLAY CHILDREN :')
        print(display_root_children)
        print('/ DISPLAY CHILDREN')'''
        #return display_root_children
    #else :
    #    print(parent + " pas dans l'index du dataframe in rameau_parents_grouped")

def jsonTo606(data, structure, formatt) :

    pred_dict = {}
    if "union" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["union"]) > 0 :
            for prediction in data["PredictionByAggregation"]["union"] :
                print("prediction union",prediction)
                pred_dict[prediction['id']] = prediction['label']
    if "intersection" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["intersection"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["intersection"] :
                print("prediction intersection",prediction)
                pred_dict[prediction['id']] = prediction['label']

    if "intersection2models" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["intersection2models"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["intersection2models"] :
                print("prediction intersection2models",prediction)
                pred_dict[prediction['id']] = prediction['label']
                
    if "intersection2models1best" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["intersection2models1best"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["intersection2models1best"] :
                print("prediction intersection2models1best",prediction)
                pred_dict[prediction['id']] = prediction['label']
                
    if "cross" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["cross"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["cross"] :
                print("prediction cross",prediction)
                pred_dict[prediction['id']] = prediction['label']
                
    if "emb" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["emb"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["emb"] :
                print("prediction emb",prediction)
                pred_dict[prediction['id']] = prediction['label']
                
    if "llm" in data["PredictionByAggregation"].keys() :
        if len(data["PredictionByAggregation"]["llm"]) > 0 :
            pred_dict = {}
            for prediction in data["PredictionByAggregation"]["llm"] :
                print("prediction llm",prediction)
                pred_dict[prediction['id']] = prediction['label']
    

    print('pred_dict : ')
    print(pred_dict)
    
    suggested_ids = list(pred_dict.keys())
    print(suggested_ids)

    if structure == 'flat' :
        display = []
        for suggested_id in suggested_ids :
            display.append(set606(suggested_id, pred_dict[suggested_id], formatt))
    
    elif structure == 'tree' :

        not_displayed = suggested_ids
        print()
        print('not_displayed : ')
        print(not_displayed)
    
        suggested_roots = getRoots(pred_dict)
        print()
        print('suggested_roots :')
        print(suggested_roots)
    
        #suggested_ancestors = rameau_ancestors_df[rameau_ancestors_df['ppn'].isin(suggested_ids)]
    
        display = []
        #print(display)
    
        for root_id in suggested_roots :
            print(root_id)
            #print(not_displayed)
            #display = display + display_roots(root_id, not_displayed, pred_dict)
            #getChildren(suggested_id, 0, not_displayed, 0)
            display_root_children = display_roots(root_id, not_displayed, pred_dict, formatt) #type List
            #print(type(display_root_children))
            print(''.join(display_root_children))
            
            display = display + display_root_children
    
        print('not_displayed : '+str(not_displayed))
        
        for not_displayed_id in not_displayed :
            print('not displayed : ')
            print(set606(not_displayed_id, pred_dict[not_displayed_id],formatt))

    #for prediction_id in list(pred_dict.keys()) :
        #print(set606(prediction_id, pred_dict[prediction_id]))
    print()
    print('Final :')
    return display

instruction = "Parmi la liste de mots-clés qui suit, donne-moi la sous-liste des mots-clés qui ne représentent pas le contenu du résumé"
#instruction = "Parmi la liste de mots-clés qui suit, donne-moi la sous-liste des mots-clés qui ne représentent pas le contenu du résumé, sous la même forme "

@app.get("/subject_indexation/")
async def index_subjects(Title: str, Summary: str = None, docId: str = '', models: str = None, aggregationType: str = None, vocabulary: str = 'rameau', subjectsMaxCount: int = 5, Agent: str = '', Format: str = 'json',request: Request=None):
    #print(request.headers['user-agent'])
    liste_models = ['victor3_chain', 'victor1_concept','victor1_chain','victor2']  # english model 'victor3_chain_en' not yet in this list
    liste_agregation = ['union','intersection','intersection2models','intersection2models1best', 'llm', 'lilimarlene','cross','qdrant','embbed']
    liste_models_best = [ 'victor1_concept', 'victor3_chain']
    liste_complementarymodels = ['victor2'] # pour intersection2models1best (ne garder ici que les modèles vraiment différents des bestmodels.)
    liste_models_in_param = [] #models given in the url param "models"
    #print("liste_models_in_param un debut : "+str(liste_models_in_param))
    
    
        
    result_data = {}
    response_data = {
        "DocumentID": docId,
        "PredictionByModel": {},
        "PredictionByAggregation": {}
    }

    vocabulary = vocabulary.lower()

    # Si le param "models" est vide, on ne prend que les modèles listés comme "Best"
    # Si le param "models" est *, on prend tous les modèles possibles
    if models is None:
        liste_models_in_param = [x for x in liste_models_best]
    # pour demander tous les modèles
    elif models == "*" :
        liste_models_in_param = [ x for x in liste_models]
    else:
        liste_models_in_param = [m.strip().lower() for m in models.split(',')]
    #print("liste_models_in_param après test None ou * : "+str(liste_models_in_param))
    #print("liste_models_in_param après test None ou * : "+str(liste_models_in_param))
        
    if Summary is None :
        Summary = ""
    #print(Summary)
   
    # Si le titre et/ou le résumé ne sont pas en français, on ajoute le modèle int float/multilingual-e5-large (même si seulement calculé sur de l'anglais))
    if get_lang(Title) != "fr"  :
        # ajout d'un modèle multilingue
        liste_models_in_param.insert(0, "victor3_chain_en") # ajout de victor3_chain_en 
        liste_models.insert(0, "victor3_chain_en") # ajout de victor3_chain_en
        liste_models_best.insert(0, "victor3_chain_en") # ajout de victor3_chain_en
        # retrait d'un modèle inadapté : plutôt le faire côté client
        '''
        liste_models_in_param.remove("victor1_concept")
        liste_models.remove("victor1_concept")
        liste_models_best.remove("victor1_concept")
        '''
        #print("liste_models_in_param après test langue : "+str(liste_models_in_param))
    elif Summary != "" and Summary != None :
        #print("Summary pas nul")
        #print("get_lang(Summary) :"+get_lang(Summary)+"fin")
        if get_lang(Summary) != "fr" :
            #print("Summary pas fr")
            # ajout d'un modèle multilingue
            liste_models_in_param.insert(0, "victor3_chain_en") # ajout de victor3_chain_en 
            liste_models.insert(0, "victor3_chain_en") # ajout de victor3_chain_en
            liste_models_best.insert(0, "victor3_chain_en") # ajout de victor3_chain_en
            # retrait d'un modèle inadapté : plutôt le faire côté client
            #liste_models_in_param.remove("victor1_concept")
            #liste_models.remove("victor1_concept")
            #liste_models_best.remove("victor1_concept")
    
    if liste_models_in_param:
        for model in liste_models_in_param :
            if model.strip() not in liste_models:
                        raise HTTPException(status_code=404, detail=f"Le modèle '{model.strip()}' n'est pas dans la liste des modèles disponibles.Modèles disponibles : {', '.join(liste_models)}")

   #xx
    RCRs = ["991279901","341720001","RCR","040702201","040702202","041922301","050612101","060832301","060835201","081052203","103872101","103872201","103872202","110692201","130010001","130012101","130012102","130012103","130012104","130012105","130012201","130012202","130012203","130012204","130012205","130012206","130012207","130012210","130012211","130012212","130012213","130012214","130012215","130012216","130012218","130012219","130012220","130012221","130012222","130012223","130012224","130012225","130012226","130012227","130012228","130012229","130012230","130012231","130012232","130012233","130012234","130012235","130012236","130012303","130012305","130015206","130019801","130019802","130042201","130042202","130052201","130282201","130552101","130552102","130552103","130552104","130552105","130552106","130552107","130552108","130552109","130552205","130552206","130552207","130552208","130552209","130552210","130552316","130555104","130555208","130555403","130559901","130559902","130559903","130559904","132032201","132132301","132135201","132139901","141182304","212312306","221132206","221132207","222772103","222782101","222782201","241725201","263622101","263622201","263622202","263622203","263622302","290395201","301892101","301892102","301892103","301892104","301892105","301892201","301899901","340322101","340322102","341720002","341722101","341722102","341722103","341722104","341722105","341722106","341722108","341722109","341722110","341722111","341722112","341722113","341722204","341722206","341722207","341722208","341722209","341722210","341722211","341722212","341722213","341722214","341722215","341722216","341722217","341722218","341722219","341722220","341722221","341722222","341722223","341722224","341722225","341722226","341722227","341722228","341722229","341722230","341722231","341722232","341722233","341722234","341722235","341722236","341722237","341722238","341722239","341722240","341722241","341722242","341722243","341722244","341722245","341722246","341722247","341722295","341722296","341722298","341722299","341722321","341723001","341725106","341725107","341729801","341729802","341729901","343012101","350472301","350935201","352380001","352380002","352382101","352382102","352382103","352382104","352382105","352382106","352382122","352382126","352382136","352382139","352382209","352382210","352382212","352382213","352382215","352382217","352382221","352382227","352382230","352382233","352382237","352382238","352382239","352382240","352382241","352382242","352382243","352382244","352382245","352382246","352382248","352382249","352382250","352382251","352382339","352385136","352385201","352389801","352389902","361455201","372610001","372610011","372612101","372612102","372612103","372612104","372612201","372612203","372612204","372612205","372612206","372612208","372612209","372612210","372612211","372612212","372612213","372612214","372612215","372612216","372612217","372612218","372612302","372615206","372615207","372615209","372619801","381512201","381852101","381852202","381852203","381852205","381852206","381852207","381852208","381852209","381852210","381852301","381852305","381852306","384210001","384212101","384212102","384212103","384212202","384212203","384212205","384212207","384212208","384212209","384212213","384212214","384212218","384212221","384212223","384212224","384212226","384212227","384212229","384212233","384212237","384212238","384212301","384212302","384212304","384215102","384215204","384219801","384219901","384219902","385162101","385442201","410182101","480952201","511082201","514540001","514542101","514542102","514542103","514542104","514542217","514542218","514542304","514549801","514549901","521212201","543952311","661362201","730652301","740102302","740422301","751050006","751050009","751052119","751052304","751052305","751052306","751052307","751052308","751052309","751052310","751052311","751052312","751052313","751052328","751052337","751055103","751055104","751055105","751055106","751055206","751055207","751055208","751055209","751055214","751055215","751055221","751055226","751055229","751055232","751055233","751059804","751059807","751059902","751059911","751070003","751072303","751079802","751079901","751122301","751132301","751162302","763512303","781585201","840072203","841275201","861942305","911145101","991270001","991272301","991272302","991272303","991279801"]

    if Agent =="" :
        return HTMLResponse(content=f"Init avec agent null ok.", status_code=200)
    
    if Agent not in RCRs :
        return HTMLResponse(content=f"Votre bibliothèque ne peut accéder à ce service réservé aux établissements testeurs.", status_code=200)
        #raise HTTPException(status_code=400, detail=f"Votre bibliothèque ne peut accéder à ce service réservé aux établissements testeurs.")
    
    if aggregationType is not None:
        cleaned_types = aggregationType.lower()
        if aggregationType == "*" :
            types_list = liste_agregation
        elif "," in cleaned_types:
            types_list = cleaned_types.split(',')
        else:
            types_list = [cleaned_types]
        
        types_propres = [element.strip() for element in types_list]
    print("types_propres",types_propres)   
    
    if aggregationType:
        for aggregation in types_propres:
            if aggregation.strip() not in liste_agregation:
                        raise HTTPException(status_code=404, detail=f"Le type '{aggregation.strip()}' n'est pas dans la liste des types d'agrégation disponibles. Agrégations disponibles : {', '.join(liste_agregation)}")
    
    if vocabulary == 'rameau':
        for model_name in liste_models_in_param:
            # Marqueur de temps pour chaque modèle
            if model_name == 'victor1_concept':
                start_time_model = time.time() 
                result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder1, 'concepts_allMin_only_mono', subjectsMaxCount)
                end_time_model = time.time()
            elif model_name == 'victor1_chain':
                start_time_model = time.time() 
                result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder1, 'concepts_allMin_only_mono', subjectsMaxCount)
                end_time_model = time.time()
            elif model_name == 'victor2':
                start_time_model = time.time() 
                result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder2, 'concepts_distiluse_only_mono', subjectsMaxCount)
                end_time_model = time.time()
            elif model_name == 'victor3_chain':
                start_time_model = time.time() 
                result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder3, 'concepts_e5-large_only_mono', subjectsMaxCount)
                end_time_model = time.time()
            elif model_name == 'victor3_chain_en':
                start_time_model = time.time() 
                result_data[model_name], _ = process_embedding_qdrant(Title, Summary, Qdrant_Client, encoder3, 'concepts_e5-large_only_mono', subjectsMaxCount)
                end_time_model = time.time()
            
            # Limiter le nombre de résultats si nécessaire
            if subjectsMaxCount is not None and subjectsMaxCount > 0:
                for key in result_data:
                    result_data[key] = result_data[key][:subjectsMaxCount]
            # Ajoutez le temps d'exécution pour ce modèle
           
            response_data["PredictionByModel"][model_name] = {
                "Result": result_data[model_name],
                "ResponseTime": f"{round(end_time_model - start_time_model, 2)} secondes"
            }

# UNION
    model_results = []
    model_results_best = []

    for model_name, result in result_data.items():
        model_results.append(result_data[model_name])
    union_labels_ids = set()
    for sublist in model_results:
        for item in sublist:
            label_id_tuple = (item['label'], item['id'])
            union_labels_ids.add(label_id_tuple)
    union_list = [{"label": label, "id": id} for label, id in union_labels_ids]
    print("union_list",union_list)

    # Union avec seulement les modèles best, pour passer au llm notamment (exclure les modèles non best, qui ne servent que pour l'intersection par ex)
    for model_name, result in result_data.items():
        if model_name in liste_models_best :
            model_results_best.append(result_data[model_name])
    union_labels_ids_best = set()
    for sublist in model_results_best:
        for item in sublist:
            label_id_tuple = (item['label'], item['id'])
            union_labels_ids_best.add(label_id_tuple)
    union_list_best = [{"label": label, "id": id} for label, id in union_labels_ids_best]
    print("union_list_best",union_list_best)

    
# INTERSECTION
    model_results = []

    # Ajouter les résultats du modèle à la liste des résultats
    for model_name, result in result_data.items():
        model_results.append(result_data[model_name])

    # Initialiser l'intersection avec le premier ensemble de labels et IDs
    intersection_labels_ids = set((item['label'], item['id']) for item in model_results[0])

    # Parcourir les sous-listes restantes et trouver l'intersection
    for sublist in model_results[1:]:
        sublist_labels_ids = set((item['label'], item['id']) for item in sublist)
        intersection_labels_ids &= sublist_labels_ids

    # Convertir le set en une liste de dictionnaires
    intersection_list = [{"label": label, "id": id} for label, id in intersection_labels_ids]
    print("intersection_list",intersection_list)
    
# INTERSECTION2MODELS
   
    resultats_2models = set()

    for model_name, result in result_data.items():
        ensemble_resultat = set()
        for item in result:
            element = item['label']
            element_id = item['id']
            present_dans_autre_ensemble = False

            for autre_model, autre_resultat in result_data.items():
                if autre_model != model_name:
                    for autre_item in autre_resultat:
                        if element == autre_item['label']:
                            present_dans_autre_ensemble = True
                            break
                if present_dans_autre_ensemble:
                    break

            if present_dans_autre_ensemble:
                ensemble_resultat.add((element, element_id))

        resultats_2models.update(ensemble_resultat)

    resultats_2models_list = [{"label": label, "id": id} for label, id in resultats_2models]
    print("resultats_2models",resultats_2models_list)
    
# INTERSECTION2MODELS1BEST
    resultats_best = set()
    for model_name, result in result_data.items():
        if model_name in liste_models_best:
            ensemble_resultat = set()
            for item in result:
                element = item['label']
                element_id = item['id']
                present_dans_autre_ensemble = False

                # Vérifier si l'élément est présent dans les autres modèles
                for autre_model, autre_resultat in result_data.items():
                    if autre_model in liste_complementarymodels:
                        for autre_item in autre_resultat:
                            if element == autre_item['label']:
                                present_dans_autre_ensemble = True
                                break
                    if present_dans_autre_ensemble:
                        break

                # Ajouter l'élément au set s'il est présent dans un autre modèle
                if present_dans_autre_ensemble:
                    ensemble_resultat.add((element, element_id))

            # Mettre à jour le set global avec les résultats uniques
            resultats_best.update(ensemble_resultat)

    # Convertir le set en une liste de dictionnaires
    resultats_best_list = [{"label": label, "id": id} for label, id in resultats_best]
    print("resultats_best_list",resultats_best_list)
    # AGREGATION LLM 
    
    union_labels_best = [x['label'] for x in union_list_best]
    union_labels_a = [x['label'] for x in union_list]
    #print(union_labels_best)                     
    list_a_traiter=[]
    if aggregationType:
        aggregated_predictions = {}

        if "union" in types_propres:
            print("ajout union")
            aggregated_predictions["union"] = union_list
            list_a_traiter=union_list

        if "intersection" in types_propres:
            aggregated_predictions["intersection"] = intersection_list
            list_a_traiter=intersection_list

        if "intersection2models" in types_propres:
            aggregated_predictions["intersection2models"] = resultats_2models_list
            list_a_traiter=resultats_2models_list

        if "intersection2models1best" in types_propres:
            aggregated_predictions["intersection2Models1Best"] = resultats_best_list
            list_a_traiter=resultats_best_list
            
        labels_a_traiter = [x['label'] for x in list_a_traiter]
        
        if "llm" in types_propres or "lilimarlene" in types_propres:
            print("llm",types_propres)
            union_llm_list_best=[]
            union_llm_list_best = llm_postprocess(llm_select(Title+". "+Summary, ", ".join(labels_a_traiter), instruction), list_a_traiter)
            
            print("union_llm_list_best",union_llm_list_best)
            aggregated_predictions["llm"] = union_llm_list_best
            #aggregated_predictions["llm"] = resultats_2models_list
        if "cross" in types_propres:
            #print("llm")
            union_llm_list_cross=[]
            
            tmp=cross_select(Title+". "+Summary, labels_a_traiter, list_a_traiter)
            union_llm_list_cross = cross_postprocess(tmp, list_a_traiter)
            print("union_llm_list_cross",union_llm_list_cross)
            aggregated_predictions["cross"] = union_llm_list_cross
            
        if "embbed" in types_propres:
            #print("llm")
            union_llm_list_cross=[]
            
            tmp=embed_select(Title+". "+Summary, labels_a_traiter, list_a_traiter)
            union_llm_list_emb = emb_postprocess(tmp, list_a_traiter)
            print("union_llm_list_emb",union_llm_list_emb)
            aggregated_predictions["emb"] = union_llm_list_emb
            #aggregated_predictions["llm"] = resultats_2models_list

        response_data["PredictionByAggregation"] = aggregated_predictions
        print("response_data",response_data)

    # fichier de traces
    instant = datetime.datetime.now()                                
    file_path = root+"history.txt";
    with open(file_path, 'a') as file:
        try :
            file.write( "\n" + Agent + ";" + docId + ";" + Title.replace(";",",") + ";" + Summary.replace(";",",") + ";"+ str(instant))
        except Exception as e :
            print(e)

    #fichier de réponse json
    instant_clean = str(instant).replace(":","-").replace(".","-").replace(" ","-")
    os.makedirs(root+"responses", exist_ok=True)
    if docId != "" :
        file_path_response = root+"responses/"+docId + "_" + instant_clean+".json";
    else :
        file_path_response = root+"responses/"+instant_clean+".json";
    with open(file_path_response, 'a') as file_response:
        try :
            file_response.write(json.dumps(response_data))
        except Exception as e :
            print(e)
            
    page="""<!DOCTYPE html>
    <html>
        <head>
            <title>Propositions</title>
            <meta charset="utf-8" />
        </head>
        <body>"""+ ''.join(jsonTo606(response_data, 'flat', 'html')).replace("\n","<br />") +"""
            
        </body>
    </html>"""
    
    page_tree="""<!DOCTYPE html>
    <html>
        <head>
            <title>Propositions</title>
            <meta charset="utf-8" />
        </head>
        <body>"""+ ''.join(jsonTo606(response_data, 'tree', 'html')).replace("\n","<br />") +"""
            
        </body>
    </html>"""
    
    if Format =="text" :
        #return HTMLResponse(content=jsonTo606(response_data), status_code=200)
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'tree', 'text')), status_code=200)
    if Format =="text_flat" :
        #return HTMLResponse(content=jsonTo606(response_data), status_code=200)
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'flat', 'text')), status_code=200)
    if Format =="text_tree" :
        #return HTMLResponse(content=jsonTo606(response_data), status_code=200)
        return HTMLResponse(content=''.join(jsonTo606(response_data, 'tree', 'text')), status_code=200)
    if Format =="html" :
        return HTMLResponse(content=page_tree, status_code=200)
    if Format =="html_flat" :
        return HTMLResponse(content=page, status_code=200)
    if Format =="html_tree" :
        return HTMLResponse(content=page_tree, status_code=200)
    else:
        return JSONResponse(content=response_data)

    #return response_data

rameau_html = """
<!DOCTYPE html>
<html>
<body>

<h2>Rameau</h2>

<div id="rm">
<p>RCR</p>
<textarea id="rcr" name="rcr" rows="1" cols="20">
</textarea>
<p>Titre</p>
<textarea id="titre" name="titre" rows="3" cols="200">
</textarea>
<p>Résumé</p>
<textarea id="resume" name="resume" rows="3" cols="200">
</textarea>
<p>Nombre de résultats par modèle</p>
<textarea id="hitscount" name="hitscount" rows="1" cols="20">5</textarea>
<br />
<button type="button" onclick="loadRameau()">envoi</button>
<p>Rameau</p>
<div id="resu">
</resu>

</div>

<script>
function loadRameau() {
  const xhttp = new XMLHttpRequest();
  xhttp.onload = function() {
    document.getElementById("resu").innerHTML = this.responseText;
  }
  xhttp.open("GET","https://hub-rdftotal.idref.fr/subject_indexation/?docId=null&Title="+encodeURIComponent(document.getElementById("titre").value)+"&Summary="+encodeURIComponent(document.getElementById("resume").value)+"&models=victor3_chain,omi2,victor1_concept,victor2&aggregationType=intersection2models,llm&subjectsMaxCount="+encodeURIComponent(document.getElementById("hitscount").value)+"&Agent="+document.getElementById("rcr").value+"&vocabulary=rameau&Format=html"
);
  xhttp.send();
}
</script>

</body>
</html>
"""

#formulaire de depot de la these en pdf
@app.get("/rameau")
async def present_form_rameau():
    return HTMLResponse(rameau_html)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8071)