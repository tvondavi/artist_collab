#------------------ Limitations ------------------------------------
# No context awareness — negation ("not happy") isn't handled well
# Lexicon coverage — words not in the lexicon are ignored
# No sarcasm/irony detection
# Less accurate than transformer-based models (e.g., fine-tuned BERT) on nuanced text
# The underlying lexicon was crowd-annotated and may reflect biases

#---------------- When NRCLex might still be preferred ---------------------

# You need interpretability (which specific words drove the score)
# You're processing very large volumes of text and speed/cost matters
# You need the full 8-emotion + sentiment Plutchik wheel




#import libraries and modules
import sys
print(sys.executable)
import numpy as np
import pandas as pd
import json
import glob
# NRCLex for lexicon-based emotion analysis
from nrclex import NRCLex
import os
import re
import matplotlib.pyplot as plt

####----------------- Set Up Dataframe with the CSV ----------------###
df = pd.read_csv("llama_translated_file.csv")
texts = df["translated_text"].fillna("").astype(str).tolist()

####----------------- Clean the list to get the formatted titles ----------------###
def preprocess_texts_for_emotion(texts):
    processed_texts = []
    for doc in texts:
        doc = doc.strip().lower()
        doc = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\"']", "", doc)
        processed_texts.append(doc)
    return processed_texts

clean_titles = preprocess_texts_for_emotion(texts)

####----------------- Define the full NRCLex emotion + sentiment categories ----------------###
NRC_CATEGORIES = [
    'fear', 'anger', 'anticipation', 'trust',
    'surprise', 'sadness', 'joy', 'disgust',
    'positive', 'negative'
]

####----------------- Run NRCLex and collect affect frequencies ----------------###
def get_nrc_scores(text, max_length=10000):
    text = text[:max_length]
    analyzer = NRCLex()
    analyzer.load_raw_text(text)
    freqs = analyzer.affect_frequencies
    return {cat: freqs.get(cat, 0.0) for cat in NRC_CATEGORIES}

emotion_scores = [get_nrc_scores(text) for text in clean_titles]

####----------------- Update the dataframe with the emotion scores ----------------###
emotion_df = pd.DataFrame(emotion_scores, columns=NRC_CATEGORIES)

df = pd.concat([df, emotion_df], axis=1)

# Top emotion restricted to the 8 core emotions (excluding positive/negative sentiment)
EMOTION_ONLY = NRC_CATEGORIES[:8]
emotion_only_df = emotion_df[EMOTION_ONLY]

df['top_emotion'] = emotion_only_df.idxmax(axis=1)
df['top_emotion_score'] = emotion_only_df.max(axis=1)

####----------------- Handle texts with no recognizable emotion words ----------------###
# If all emotion scores are 0, top_emotion is ambiguous — flag these rows
df['top_emotion'] = df.apply(
    lambda row: 'none' if row[EMOTION_ONLY].sum() == 0 else row['top_emotion'],
    axis=1
)

####----------------- Create a new CSV from the newly updated dataframe ----------------###
df.to_csv("title_emotions_nrc.csv", index=False)