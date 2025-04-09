# 1 set LLM Mistral 7B
import os

from langchainHuggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


#  SET UP ENVIRONMENT VARIABLES
HF_TOKEN = os.environ.get('HF_TOKEN')
HugFace_repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'


def load_llm(HugFace_repo_id):
  try:
    llm = HuggingFaceEndpoint(
      repo_id=HugFace_repo_id,
      model_kwargs={
        "temperature": 0.4,
        "max_new_tokens": 512,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "num_return_sequences": 1,
      },
      huggingfacehub_api_token=HF_TOKEN
    )
    return llm
  except Exception as e:
    print(f"Error loading LLM: {e}")
    return None

