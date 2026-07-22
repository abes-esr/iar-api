
# ============================================================
# 
# <h1>Prédire les cocnepts Rameau avec une base de vecteurs</h1>
# 
# 
# * Pour chaque ligne de l'export, tu calcules le vecteur du titre-résumé. Rappel : une ligne peut posséder plusieurs vedettes rameau  (une vedette, c'est une zone 606, avec ou sans subdivision, cad avec un ou plusieurs ppn)
# * Puis, pour chaque vedette rameau, tu agrèges les titres+résumés associés (GROUP BY), et tu fais la moyenne de leur vecteur (AGG(MEAN)
# * Tu obtiens donc un vecteur par vedette rameau, qui "représente" toute la richesse contenue dans les titres et résumés des notices biblio ayant cette vedette. Appelons-ça ta grosse matrice Rameau++
# * Ensuite, avec ce modèle, tu peux prédire les vedettes à associer à une nouvelle notice. Pour ce faire, tu vectorises le titre+résumé de la nouvelle notice et tu cherches le vecteur le plus proche de ta grosse matrice Rameau++
# 

# ============================================================

# Note: Les installations de paquets sont à lancer manuellement si nécessaire :
# sys.executable -m pip install pinecone-client sentence-transformers datasets



import pandas as pd
import numpy as np
import simplemma
import nltk
from texthero import preprocessing as preprocessing2
import texthero as hero
import pickle


def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return ' '.join(word for word in [simplemma.lemmatize(w, lang='fr') for w in w_tokenizer.tokenize(text)])

import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="VOTRE_CLE_PINECONE",
    environment="us-west4-gcp"  # find next to API key in console
)

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# pour re-indexer il faut au prealable supprimer l'index sur l'interface d'admin
index_name = 'extreme-ml'

# check if the extreme-ml index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=384,
        
        metric="cosine"
    )

# connect to extreme-ml index we created
index = pinecone.Index(index_name)

#chargemnt fichier d'entrainement

#df1 = pd.read_csv('export_picone.csv', sep='\t', skiprows=[12635,14570,20335,20655,32592,32975,32976,33310,50886,58789,58841,
#59676,63232,67341,72847,79427,80517,88343])
df1 = pd.read_csv('working_data_sans_dewey.csv', sep=',')

a = ['000308838','003632806','047450037','058296182','059911174','067313493','070072973','076503909','076986152','077463560','077880560','086077368','103220844','120997703','126056536','137422091','146527313','146979168','147294509','157175065','159761875','162374631','163093741','166049921','166278351','176553460','181543656','182508188','183201523','191351059','191415782','191552208','192576445','192816969','196109590','196122708','197101267','198384122','198388810','200050818','200404342','201602423','219465118','221455183','223827959','225697122','227065069','230373828','230756883','231055099','231860838','232821909','234544538','235109614','235130273','235280011','235755265','236616587','236660640','237131560','237156989','241152550','243051646','248194305','248590413','248915053','248944479','249549492','252457234','252816528','254162525','254992609','255264887','257349006','257504990','257936432','258740043','261199609','261643614','262267888','262760606','263439038','263487784','263926400','265476585','266197809','267884575','268799458','268924759','00094758X','05224170X','11707764X','18171681X','19580547X','23097368X','23690454X','24155859X','25561280X','26117309X','26753177X']
df1.info(verbose = False)
exclus=df1[df1['PPN'].isin(a)]
df1 = df1[~df1['PPN'].isin(a)]

df1.info(verbose = False)

mask = np.random.rand(len(df1)) < 0.81
training_data = df1[mask]
training_data.info(verbose = False)
training_data['PPN'].to_csv('data/training_data.csv',index=False)
testing_data = df1[~mask]
testing_data.info(verbose = False)
testing_data = testing_data.append(exclus, ignore_index=True)
testing_data.info(verbose = False)
testing_data['PPN'].to_csv('data/test_data.csv',index=False)
training_data["RESUME"] = training_data['TITRE'].astype(str) +", "+ training_data["RESUME"]
df=training_data.drop(columns=['PPN','TITRE','DEWEY'])

df['RAMEAU']=df['RAMEAU'].str.replace('_','#u#', regex=True).replace('"','#d#', regex=True).replace('\'','#c#', regex=True).replace(' ','_', regex=True)

#df.drop(index=df.index[89000:],axis=0, inplace=True)
#df.reset_index(drop=True, inplace=True)
#tronque
df['RESUME'] = df['RESUME'].str.slice(0,1000)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
#lemmatization
df['RESUME'] = df['RESUME'].apply(lemmatize_text)

df.reset_index(drop=True, inplace=True)


testing_data.to_csv('testing_data_full.csv',index=False)


#NETTOYAGE
clean_pipeline = [preprocessing2.fillna,
                   preprocessing2.lowercase,
                   preprocessing2.remove_whitespace,
                   preprocessing2.remove_diacritics
                   
                  ]
df['RESUME'] = hero.clean(df['RESUME'], clean_pipeline)

df.head(5)


df['RAMEAU'] = df['RAMEAU'].str.split(';')
df.head(5)

#ne marche pas apres car type string suppression de ce qui suit -- 
#import re
#ch = '_--_'
#pattern  = ch + ".*"
#df["RAMEAU"] = df["RAMEAU"].apply(lambda x: re.sub(pattern, '', str(x) ))
#df.head(5)


#inutile
#creation jeux de test (les 460 derniers)
df_test=df.copy()
df_test.drop(index=df_test.index[:89000],axis=0, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_test
#print (len(df_test))

#inutile
#creation jeu de vectorization, les 89000 premiers
#suppression records de test
#index_list = list(range(9000, 9416))
#df.drop(df.index[index_list], inplace =True)
df.drop(index=df.index[89000:],axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
print (len(df))
df

#chargement model
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model from huggingface
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device=device
)
model

df

#creation des vecteurs
import pandas as pd

# Create embeddings
encoded_articles = model.encode(df['RESUME'].tolist(), show_progress_bar=True)
# add the embeddings to our dataframe
df['content_vector'] = pd.Series(encoded_articles.tolist())

#agregation
import numpy as np


df_explode = df.explode('RAMEAU')
#df_explode.head(5)

label_vectors = df_explode.groupby('RAMEAU').agg(mean=('content_vector', lambda x: np.vstack(x).mean(axis=0).tolist()))
label_vectors['target'] = label_vectors.index
label_vectors.columns = ['content_vector', 'label']

label_vectors.sample(10)



label_vectors.head(100)

#sauvegarde des vecteur car long a produire
import pickle
label_vectors.to_pickle('cb_test_class_v2_vector_label.pkl') 
#label_vectors = pd.read_pickle('cb_test_class_v2_vector_label.pkl')

print(label_vectors.size)



#alimentation index picone
from tqdm.auto import tqdm

# we will use batches of 256
batch_size = 256

for i in tqdm(range(0, len(label_vectors), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(label_vectors))
    # extract batch
    batch = label_vectors.iloc[i:i_end]
    # select embeddings for batch
    emb = batch["content_vector"].tolist()
    # get metadata
    meta = [{"label": l} for l in batch["label"]]
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

to_upsert

# check that we have all vectors in index
index.describe_index_stats()

from pprint import pprint
def select_test_article(index):
    print("choisi un element du fichier de  test:")
    # select the article associated with index from test split
    article = df_test.iloc[index]
    # print test article data
    data = {"RESUME": article.RESUME[:1000],  "Original Labels": list(article.RAMEAU)}
    pprint(data)
    return article
def query_pinecone(article, top_k=3):
    print("predict rameau:")
    # Create embeddings for test articles
    xq = model.encode(article.RESUME).tolist()
    # query pinecone for labels
    results = index.query(xq, top_k=top_k, include_metadata=True)
    print(results)
    # select only the labels from result and print
    labels = [res["metadata"]["label"] for res in results.matches]
    pprint({"Predicted Labels": labels})

#test sur le sixieme record du jeu de test
article = select_test_article(0)

#interrogation db vectorized pour ce record
query_pinecone(article, top_k=10)

#generation fichier excel sur evaluation

#import pandas as pd
import numpy as np
import simplemma
import nltk
from texthero import preprocessing as preprocessing2
import texthero as hero
import pickle
import pandas as pd
import numpy as np




def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return ' '.join(word for word in [simplemma.lemmatize(w, lang='fr') for w in w_tokenizer.tokenize(text)])

import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="VOTRE_CLE_PINECONE",
    environment="us-west4-gcp"  # find next to API key in console
)

def predict(text):
    xq = model.encode(lemmatize_text(text)).tolist()
    results = index.query(xq, top_k=10, include_metadata=True)
    return results

#chargement model
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model from huggingface
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device=device
)

index = pinecone.Index('extreme-ml')
# faut jour les entetes et le connection a la base vecteurs
#df1 = pd.read_csv('export_picone_100.csv', sep='\t', skiprows=[12635,14570,20335,20655,32592,32975,32976,33310,50886,58789,58841,59676,63232,67341,72847,79427,80517,88343])
df1=testing_data.head(100).copy()
#df1=testing_data.copy()
df1["RESUME"] = df1['TITRE'].astype(str) +", "+ df1["RESUME"]
#-df=df1.drop(columns=['PPN','TITRE','DEWEY'])
df_test=df1
df_test['RAMEAU']=df_test['RAMEAU'].str.replace('_','#u#', regex=True).replace('"','#d#', regex=True).replace('\'','#c#', regex=True).replace(' ','_', regex=True)
#tronque
df_test['RESUME'] = df_test['RESUME'].str.slice(0,1000)
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
#lemmatization
df_test['RESUME'] = df_test['RESUME'].apply(lemmatize_text)
df_test['RAMEAU'] = df_test['RAMEAU'].str.split(';')
#df_test=df.copy()
#df_test.drop(index=df_test.index[:89200],axis=0, inplace=True)
#df_test.reset_index(drop=True, inplace=True)
#predictions
df_test['PREDICT'] = df_test['RESUME'].apply(predict)
#df_test['PREDICT'].head()
#df_test.to_csv('test_cbd.csv', sep='\t', encoding='utf-8')
df_test.to_csv('test_cbd_100_2.csv',index=False)
df_test

import json
import csv
import pandas
import ast

df_res=pd.DataFrame()

def extract_json(ppn,text):
    #text = ast.literal_eval(json.dumps(text))
    global df_res
    #print(text)
    x = json.loads(text)
    load_items = []
    
    for item in x["matches"]:
        #print(item["metadata"]["label"])
        load_items.append({'ppn': ppn,'score': item['score'],'label': item["metadata"]["label"]  })
            #load_items.append({'metadata': item2['label'] })
    z = pd.DataFrame(load_items)
    df_res=df_res.append(load_items, ignore_index=True)
    return z
df_test2 = df_test[['PPN', 'PREDICT']]
df_test2['PREDICT'] = df_test2['PREDICT'].astype(str).replace('\'', '"', regex=True).replace('\n', '', regex=True).replace('             ', ' ', regex=True)
#df_test2['PREDICT'] = df_test2['PREDICT'].astype(str).replace('""', '#db#', regex=True).replace('"', '#dc#', regex=True).replace('#db#', '\'', regex=True).replace('#dc#', '\'', regex=True).replace('\"', '#df#', regex=True).replace('\'', '"', regex=True).replace('\n', '', regex=True).replace('             ', ' ', regex=True)



df_test2.apply(lambda x: extract_json(x.PPN, x.PREDICT), axis=1)
df_res['label']=df_res['label'].replace('#u#','_', regex=True).replace('#d#','"', regex=True).replace('#c#','\'', regex=True).replace('_',' ', regex=True)
df_res.to_csv('test_cbd_100_3.csv',index=False)

  

df_res.head()

# test sur une entree
# a jouer une fois
#import pandas as pd
import numpy as np
import simplemma
import nltk
from texthero import preprocessing as preprocessing2
import texthero as hero
import pickle



def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    return ' '.join(word for word in [simplemma.lemmatize(w, lang='fr') for w in w_tokenizer.tokenize(text)])

import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="VOTRE_CLE_PINECONE",
    environment="us-west4-gcp"  # find next to API key in console
)

def predict(text):
    xq = model.encode(lemmatize_text(text)).tolist()
    results = index.query(xq, top_k=100, include_metadata=True)
    return results

#chargement model
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model from huggingface
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device=device
)

index = pinecone.Index('extreme-ml')







predict('le dessin être un voyage : carnet de jean Léonard, le dizaine de carnet de voyages, rempli à le gré un années, temoignent de l inlassable curiosité et de le gourmandise intellectuel avec  laquelles jean Léonard aura retenir le beauté de le monde à le  pointe de le crayon. pour qui être passionné par le dessin,  feuilleter ce centaine de page être un contentement sans égal. y   être fixé le ligne un paysage contemplés, le ombre un ruer  arpentées, le contour un architecture visitées. rempart   d Essaouira, grand muraille de Chine, jardin de le Daisen-In à  kyoto, temple de Karnak, Skyline de New-York, Palais un filateur à  Ahmedabad, ruiner de Mycènes, panthéon romain, bord de mer en  bretagne...')



