# Note: Les installations de paquets sont à lancer manuellement si nécessaire :
# sys.executable -m pip install pinecone-client sentence-transformers datasets



import pandas as pd
import numpy as np

import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="VOTRE_CLE_PINECONE",
    environment="us-west4-gcp"  # find next to API key in console
)

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
df1 = pd.read_csv('export_rameau.csv') 
df=df1.drop(columns=['PPN'])
df['RAMEAU']=df['RAMEAU'].str.replace(' ','_')


df.head(20)

df = (df.groupby(['TITRE'])
      .agg({'RAMEAU': lambda x: x.tolist()})
      .reset_index())
df.head()

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

import pandas as pd

# Create embeddings
encoded_articles = model.encode(df['TITRE'].tolist(), show_progress_bar=True)
# add the embeddings to our dataframe
df['content_vector'] = pd.Series(encoded_articles.tolist())

import numpy as np

# Explode the target indicator column
df_explode = df.explode('RAMEAU')
#df_explode.head(5)
# Group by label and define a unique vector for each label
label_vectors = df_explode.groupby('RAMEAU').agg(mean=('content_vector', lambda x: np.vstack(x).mean(axis=0).tolist()))
label_vectors['target'] = label_vectors.index
label_vectors.columns = ['content_vector', 'label']

label_vectors.sample(10)



print(label_vectors.size)



from tqdm.auto import tqdm
#alimentation index 
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

# check that we have all vectors in index
index.describe_index_stats()

#chargement fichier de test
df_test = pd.read_csv('data/test_rameau_export.csv') 
df_test=df_test.drop(columns=['PPN'])
df_test['RAMEAU']=df_test['RAMEAU'].str.replace(' ','_')
df_test = (df_test.groupby(['TITRE'])
      .agg({'RAMEAU': lambda x: x.tolist()})
      .reset_index())
df_test.head()

from pprint import pprint
def select_test_article(index):
    print("choisi un element du fichier de  test:")
    # select the article associated with index from test split
    article = df_test.iloc[index]
    # print test article data
    data = {"Titre": article.TITRE[:1000],  "Original Labels": list(article.RAMEAU)}
    pprint(data)
    return article
def query_pinecone(article, top_k=3):
    print("predict rameau:")
    # Create embeddings for test articles
    xq = model.encode(article.TITRE).tolist()
    # query pinecone for labels
    results = index.query(xq, top_k=top_k, include_metadata=True)
    print(results)
    # select only the labels from result and print
    labels = [res["metadata"]["label"] for res in results.matches]
    pprint({"Predicted Labels": labels})

article = select_test_article(2)

query_pinecone(article, top_k=10)



