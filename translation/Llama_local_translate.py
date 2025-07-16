import concurrent.futures
import pandas as pd
from tqdm import tqdm  # Progress bar
import os
import re  # For removing "<think>" sections
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load CSV data
df = pd.read_csv("w_inscription.csv")
column_to_translate = "inscription text"

# Set up LLM
local_llm = "llama3.1:latest"
llm = ChatOllama(model=local_llm, temperature=0)
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Translate the following Chinese character text to English: {text}"
)
translation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=False)

# # Function to remove "<think>" reasoning sections
# def clean_translation(text):
#     return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Function to translate with a timeout
def safe_translate(text, timeout=75):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(translation_chain.predict, text=text)
        try:
            result = future.result(timeout=timeout)  # Enforce timeout
            return result
        except concurrent.futures.TimeoutError:
            print("Translation Timeout")
            return "[Translation Timeout]"

# File to save translated results
output_file = "llama_translated_file.csv"
batch_size = 10  # Adjust based on system performance

# Check if output file already exists
file_exists = os.path.isfile(output_file)

# Open progress bar
with tqdm(total=len(df), desc="Translating", unit="entry") as pbar:
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()  # Get batch
        batch["translated_text"] = batch[column_to_translate].apply(safe_translate)

        # Append batch to CSV file
        batch.to_csv(output_file, mode='a', index=False, header=not file_exists)
        
        # After first write, ensure future writes don't include headers
        file_exists = True  
        
        # Update progress bar
        pbar.update(len(batch))

print("Translation completed and saved continuously!")