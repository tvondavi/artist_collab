#import libraries and modules
import numpy as np
import pandas as pd
import json
import glob

#gensim library content for LDA
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Bigrams / trigrams
from gensim.models.phrases import Phrases, Phraser

from collections import defaultdict

#spacy libraries
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os
#visualizations
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

df = pd.read_csv("llama_translated_file.csv")

texts = df["translated_text"].fillna("").astype(str).tolist()

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess_texts(texts):
    processed_texts = []
    for doc in texts:
        doc = nlp(doc.lower())
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha
            and token.lemma_ not in STOP_WORDS
            and len(token) > 2
        ]
        processed_texts.append(tokens)
    return processed_texts


text_tokens = preprocess_texts(texts)


# Count word frequencies
word_freq = defaultdict(int)
for text in text_tokens:
    for token in text:
        word_freq[token] += 1

# Remove words that appear only once
text_tokens = [
    [token for token in text if word_freq[token] > 1]
    for text in text_tokens
]

bigram = Phrases(text_tokens, min_count=5, threshold=100)
bigram_mod = Phraser(bigram)

text_tokens = [bigram_mod[text] for text in text_tokens]

dictionary = corpora.Dictionary(text_tokens)

# Filter extremes
dictionary.filter_extremes(
    no_below=5,   # min document frequency
    no_above=0.5 # max document frequency
)

corpus = [dictionary.doc2bow(text) for text in text_tokens]

print("up to coherence score achieved")
coherence_scores = []
for k in range(5, 21):
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                   id2word=dictionary,
                   num_topics=k,
                   random_state=42,
                   passes=10,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)
    
    cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='c_v')
    score = cm.get_coherence()
    coherence_scores.append((k, score))
    print(f"Topics: {k} \t Coherence: {score:.4f}")



x, y = zip(*coherence_scores)
plt.plot(x, y, marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (c_v)")
plt.title("LDA Topic Coherence")
plt.show()
