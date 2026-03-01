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

#This will likely need to be accessible for the cluster to perform the work. Something likely will need to be downloaded.
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


df = pd.read_csv("translated_inscriptions_full.csv")

###############--------CHANGE------------####################
texts = df["translated_text"].fillna("").astype(str).tolist()
###############--------DEPENDS on CSV Title-------############

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


import matplotlib.pyplot as plt
x, y = zip(*coherence_scores)
plt.plot(x, y, marker='o')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (u_mass)")
plt.title("LDA Topic Coherence")
plt.show()


#This will identify the best coherence score and then create a topic model. This might need to be adjusted if the best coherence is for only 5 topics or fewer.
best_k, best_score = max(coherence_scores, key=lambda x: x[1])

print(f"Best number of topics: {best_k}")
print(f"Best c_v coherence: {best_score:.4f}")


best_lda = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=best_k,
    random_state=42,
    passes=10,
    chunksize=100,
    alpha='auto',
    per_word_topics=True
)

lda_vis = gensimvis.prepare(
    best_lda,
    corpus,
    dictionary,
    sort_topics=False
)

# Show inline
pyLDAvis.display(lda_vis)

# Also save to file
pyLDAvis.save_html(lda_vis, "lda_inscription_visualization.html")


###### Now we will create the csvs of information based on the topic distribution by the LDA ######
best_lda_model = best_lda

def export_topic_words(lda_model, num_words=30, output_path="topic_top_words.csv"):
    """
    Exports top words per topic with their probability scores.
    
    Args:
        lda_model: your trained gensim LDA model
        num_words: how many top words to extract per topic
        output_path: where to save the CSV
    """
    rows = []
    
    for topic_id in range(lda_model.num_topics):
        # get_topic_terms returns (word_id, probability) tuples
        top_terms = lda_model.get_topic_terms(topic_id, topn=num_words)
        
        for rank, (word_id, prob) in enumerate(top_terms, start=1):
            word = lda_model.id2word[word_id]
            rows.append({
                "topic": topic_id+1, #1-indexed for readability
                "rank": rank,
                "word": word,
                "probability": round(prob, 6),
                "log_probability": round(np.log(prob), 4)  # useful for comparing small values
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

topic_df = export_topic_words(best_lda_model, num_words=30)
topic_df.head(40)  # preview first two topics

def relevance_score(prob_word_given_topic, prob_word_in_corpus, lambda_=0.6):
    """
    The relevance metric from Sievert & Shirley (2014) used by pyLDAvis.
    lambda=0.6 is the empirically recommended default.
    """
    return lambda_ * np.log(prob_word_given_topic) + (1 - lambda_) * np.log(prob_word_given_topic / prob_word_in_corpus)

# Get marginal word probabilities across the corpus
word_counts = np.zeros(len(best_lda_model.id2word))
for bow_doc in corpus:
    for word_id, count in bow_doc:
        word_counts[word_id] += count

p_word = word_counts / word_counts.sum()  # marginal probability of each word

# Rerun export with relevance added
rows = []
for topic_id in range(best_lda_model.num_topics):
    top_terms = best_lda_model.get_topic_terms(topic_id, topn=20)
    for rank, (word_id, prob) in enumerate(top_terms, start=1):
        word = best_lda_model.id2word[word_id]
        rel = relevance_score(prob, p_word[word_id])
        rows.append({
            "topic": topic_id,
            "rank": rank,
            "word": word,
            "probability": round(prob, 6),
            "log_probability": round(np.log(prob), 4),
            "relevance_score": round(rel, 4)
        })

df = pd.DataFrame(rows)
df.to_csv("Inscription_topic_words_w_relevance.csv", index=False)


### To distribute Artwork Inscriptions by Topics #####
# Number of topics
num_topics = best_lda.num_topics  

topic_distributions = []
dominant_topics = []
dominant_probs = []

for doc_bow in corpus:
    doc_topics = best_lda.get_document_topics(doc_bow, minimum_probability=0)

    # Ensure all topics are included
    topic_probs = [0] * num_topics
    for topic_id, prob in doc_topics:
        topic_probs[topic_id] = prob
    
    topic_distributions.append(topic_probs)

    # Find dominant topic
    max_topic_id = int(np.argmax(topic_probs))
    max_prob = topic_probs[max_topic_id]

    dominant_topics.append(max_topic_id + 1)   # shift to start at 1
    dominant_probs.append(max_prob)

# Convert to DataFrame
topic_df = pd.DataFrame(topic_distributions, 
                        columns=[f"Topic_{i+1}" for i in range(num_topics)])
topic_df["Dominant_Topic"] = dominant_topics
topic_df["Topic_Probability"] = dominant_probs

# Concatenate with original df
df_with_topics = pd.concat([df, topic_df], axis=1)

# Save to CSV
df_with_topics.to_csv("Incsription_LDA_with_topics.csv", index=False)
