import pandas as pd
import simplemma
import nltk
from texthero import preprocessing as preprocessing2
import texthero as hero
import omikuji



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


def predict_sentence(sentence,model,vectorizer,le):
    # Charger le modèle
    #model = omikuji.Model.load("./model_theses")
   
    model.densify_weights(0.05)
    
    # Transformer la phrase en vecteur
    feature_value_pairs = vectorizer.transform([sentence])
    
    # Prédire
    result = []
    for x in feature_value_pairs.toarray():
        for i in range(len(x)):
            result.append([i, x[i]])
    
    # Obtenir les paires étiquette-score prédites
    label_score_pairs = model.predict(result)
    
    labels = []
    scores = []
    for pair in label_score_pairs:
        labels.append(le.inverse_transform([pair[0]])[0])
        scores.append(pair[1])

    # Créer un DataFrame avec les colonnes "label" et "score"
    df = pd.DataFrame({'label': labels, 'score': scores})
    df['id'] = df['label'].str.split('#').str[1]
    df['label'] = df['label'].str.split('#').str[0]
    df = df[['label', 'id', 'score']]
    df['label'] = df['label'].str.replace('_', ' ')


    return df

def predict_omikuji(titre, resume, model, vectorizer, le):
    # Vérifie si le résumé est présent
    if resume:
        DESCR = clean_and_lemmatize(titre, resume)
    else:
        DESCR = clean_and_lemmatize(titre, '')  # Si le résumé est absent, passe une chaîne vide
    
    valeur_specifique = DESCR[0]
    chaine_de_caracteres = str(valeur_specifique)
    result = predict_sentence(chaine_de_caracteres, model, vectorizer, le)
    
    return result



