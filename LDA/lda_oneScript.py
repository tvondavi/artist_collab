#import libraries and modules
import numpy as np
import pandas as pd
import json
import glob
import jieba
#gensim library content for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#spacy libraries
import spacy
import os
#visualizations
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt


df = pd.read_csv("newLDA_name.csv")

def load_stopwords(filepath="chinese_stopwords.txt"):
    if not os.path.exists(filepath):
        # Fallback: define a basic stopword list inline
        return set(["的", "了", "在", "是", "我", "也", "就", "都", "而", "及", "与", "着", "或", "一个", "AND", "DRAGONFLY", "CRABS"])
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def load_stopwords_from_excel(excel_path, excel_columns=None):
    df = pd.read_excel(excel_path)
    if excel_columns is None:
        excel_columns = df.columns.tolist()
    words = set()
    for col in excel_columns:
        words.update(str(x).strip() for x in df[col] if pd.notnull(x))
    return words
    
stopwords = load_stopwords()

extra_stopwords = load_stopwords_from_excel("NamesStop.xlsx", excel_columns=["Chinese Name", "English Name"])

stopwords.update(extra_stopwords)

df['raw_subject'] = df['raw_subject'].fillna("").astype(str)

# Segment text using jieba + remove stopwords + filter short tokens
def preprocess_text(text, stopwords):
    tokens = jieba.lcut(text)
    return [word for word in tokens if word not in stopwords and len(word.strip()) > 1]

# Apply the preprocessing to the DataFrame
df['tokens'] = df['raw_subject'].apply(lambda x: preprocess_text(str(x), stopwords))

# Create a dictionary and a bag-of-words (BoW) corpus
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# Train the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=7,
                            random_state=42,
                            passes=10,
                            chunksize=100,
                            alpha='auto',
                            per_word_topics=True)

# Make the coherence model
coherence_model = CoherenceModel(model=lda_model,
                                  texts=df["tokens"],
                                  dictionary=dictionary,
                                  coherence='c_v'
                                )  # 'c_v' works well for topic interpretability

coherence_score = coherence_model.get_coherence()

print(f"Coherence Score: {coherence_score:.4f}")

coherence_scores = []
for k in range(5, 21):
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                   id2word=dictionary,
                   num_topics=k,
                   random_state=42,
                   passes=10,
                   iterations=100)
    
    cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    score = cm.get_coherence()
    coherence_scores.append((k, score))
    print(f"Topics: {k} \t Coherence: {score:.4f}")


x, y = zip(*coherence_scores)
plt.plot(x, y, marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (u_mass)")
plt.title("LDA Topic Coherence")
plt.show()