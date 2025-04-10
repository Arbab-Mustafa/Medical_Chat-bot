import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Model repo ID for Mistral
HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

# STEP 1 - Load LLM
def load_llm(repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.4,
            max_new_tokens=512,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            huggingfacehub_api_token=HF_TOKEN,
            task="text-generation"
        )
        return llm
    except Exception as e:
        print(f"[LLM Error] Failed to load LLM: {e}")
        return None

# STEP 2 - Custom Prompt
CUSTOM_PROMPT = '''
Use the following context to answer the question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.

Context:
{context}

Question:
{question}

Begin your answer now.
'''

def set_custom_prompt(prompt_text):
    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_text
        )
        return prompt
    except Exception as e:
        print(f"[Prompt Error] Failed to create prompt: {e}")
        return None

# STEP 3 - Load Embeddings & Vector DB
DB_PATH = 'vectorstore/db_faiss'

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # ✅ fixed typo
    )

except Exception as e:
    print(f"[Vector DB Error] {e}")
    db = None

# STEP 4 - Setup QA Chain
llm = load_llm(HUGGINGFACE_REPO_ID)
prompt = set_custom_prompt(CUSTOM_PROMPT)

if llm and db and prompt:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # STEP 5 - Query Interaction (One-time Run)
    user_query = input("\nEnter your question: ")
    
    try:
        response = qa_chain.invoke({"query": user_query})
        print("\n🧠 Response:\n", response["result"])
        # Optionally print source documents
        # print("\n📚 Source Documents:")
        # for i, doc in enumerate(response["source_documents"], 1):
        #     print(f"\nDocument {i}:\n{doc.page_content}")
    except Exception as e:
        print(f"[Query Error] {e}")

else:
    print("\n❌ Initialization failed. Please check errors above.")
