# 1.Load Raw PDF / Data 
# 2. make them into a list of chunks
# convert into Embeddings
# 3. Store them in a vector database (Pinecone, Weaviate, Chroma, etc)


from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load env
import os
from dotenv import load_dotenv
load_dotenv()

# Load environment variables from .env file
os.environ['HF_TOKEN'] = HF_TOKEN = os.getenv('HF_TOKEN')






# Load the PDF file
data_path = 'data/'
def load_pdf(file_path):
  try:
    loader =  DirectoryLoader(file_path , glob='*pdf', loader_cls=PyPDFLoader) 
    documents = loader.load()
    return documents
  except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

documents = load_pdf(file_path=data_path)
print(f"Loaded {len(documents)} documents from {data_path}")


# create chunks of text from the documents

def create_chunks(extracted_data):
   try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks
   except Exception as e:
            print(f"Error creating chunks: {e}")
            return []
   

text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} chunks from the documents")



# convert into Embeddings
def create_embeddings():
  try:
     embading_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
     return embading_model
      
  except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []
  

embeddings = create_embeddings()


# Store them in a vector database (Pinecone, Weaviate, Chroma, etc)
DB_Path = 'vectorstore/db_faiss'
def store_embeddings(chunks, embeddings):
  try:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_Path)
    return vectorstore
  except Exception as e:
        print(f"Error storing embeddings: {e}")
        return []


vectorstore = store_embeddings(text_chunks, embeddings)
print(f"Stored {len(text_chunks)} chunks in the vector store at {DB_Path}")