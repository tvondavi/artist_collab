import pandas as pd
import os
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "../LDA_attempt/newLDA_name.csv"
OUTPUT_FILE = "llama_translated_file.csv"
COLUMN = "raw_subject"
BATCH_SIZE = 50  # Larger batches for fewer writes
NUM_WORKERS = max(1, cpu_count() - 1)  # Use all but one core

# ==============================
# SETUP GLOBAL LLM (for each worker)
# ==============================
def init_llm():
    local_llm = "llama3.1:latest"
    llm = ChatOllama(model=local_llm, temperature=0)
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following Chinese text to English: {text}. Return only the translation."
    )
    chain = prompt_template | llm
    return chain

# ==============================
# WORKER FUNCTION
# ==============================
def translate_text(text):
    global chain
    try:
        result = chain.invoke({"text": text})
        return result.content.strip()
    except Exception as e:
        return f"[Error: {e}]"

def init_worker():
    global chain
    chain = init_llm()

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    total = len(df)
    file_exists = os.path.isfile(OUTPUT_FILE)

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for i in tqdm(range(0, total, BATCH_SIZE), desc="Translating", unit="batch"):
            batch = df.iloc[i:i+BATCH_SIZE].copy()
            texts = batch[COLUMN].tolist()

            # Parallel translation
            batch["translated_text"] = pool.map(translate_text, texts)

            # Write periodically
            batch.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists)
            file_exists = True

    print("Translation completed!")
