import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#name of local llm
local_llm = "deepseek-r1:8b"

# Load the CSV data
df = pd.read_csv("w_inscription.csv")

# Identify the column to translate
column_to_translate = "inscription text"

# Create a translation chain
llm = ChatOllama(model=local_llm, temperature=0)  # Or your LLM endpoint
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Translate the following text to English: {text}"
)
translation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

# Translate the text
df["translated_text"] = df[column_to_translate].apply(
    lambda x: translation_chain.predict(text=x)
)

# Save the translated data
df.to_csv("translated_file.csv", index=False)


###################### Shortening the Inscription Dataset ###########################
# valueInterest = '无'

# filtered_df = df[df['inscription text'] == valueInterest]

# new_df = df[df['inscription text'] != valueInterest].copy()

# filtered_df.to_csv("filtered_inscription.csv", index=False)
# new_df.to_csv("w_inscription.csv", index=False)