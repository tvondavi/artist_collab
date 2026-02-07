#import libraries and modules
import numpy as np
import pandas as pd
import json
import glob

#getting emotion model from transformers torch
from transformers import pipeline

#spacy libraries
import spacy
import os
import re
#visualizations
import matplotlib.pyplot as plt

####----------------- Set Up Dataframe with the CSV ----------------###
df = pd.read_csv("llama_translated_file.csv")

texts = df["translated_text"].fillna("").astype(str).tolist()

####----------------- Clean the list to get the formatted titles ----------------###
def preprocess_texts_for_emotion(texts):
    processed_texts = []
    for doc in texts:
        # Strip extra whitespace and lowercase
        doc = doc.strip().lower()
        # Optional: remove unwanted symbols, keep punctuation that conveys tone
        doc = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\-\"']", "", doc)
        processed_texts.append(doc)
    return processed_texts


clean_titles = preprocess_texts_for_emotion(texts)


####----------------- Establish Emotion Model Pipeline ----------------###
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

emotion_scores = emotion_classifier(clean_titles)


####----------------- update the dataframe with the emotion scores ----------------###
emotion_df = pd.DataFrame([
    {d['label']: d['score'] for d in result}
    for result in emotion_scores
])

df = pd.concat([df, emotion_df], axis=1)


df['top_emotion'] = emotion_df.idxmax(axis=1)
df['top_emotion_score'] = emotion_df.max(axis=1)

####----------------- Create a new CSV from the newly updated dataframe ----------------###
df.tocsv("title_emotions.csv", index=False)