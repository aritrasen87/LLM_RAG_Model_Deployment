from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter,TokenTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv('SOURCE_DATA')
        self.emb_model_path = os.getenv('EMBED_MODEL')
        self.emb_model = self.get_embedding_model(self.emb_model_path)
        self.vector_store_path = os.getenv('VECTOR_STORE')

    def load_docs(self,path:str) -> PyPDFDirectoryLoader:
        loader = PyPDFDirectoryLoader(path)
        docs = loader.load()
        return docs
    
    def get_embedding_model(self,emb_model) -> HuggingFaceBgeEmbeddings :
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
        embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=emb_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embeddings_model
    
    def split_docs(self,docs)-> TokenTextSplitter:
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        documents = text_splitter.split_documents(docs)
        return documents
    
    def populate_vector_db(self) -> None:
        # load embeddings into Chroma - need to pass docs , embedding function and path of the db

        self.doc = self.load_docs(self.pdf_folder_path)
        self.documents = self.split_docs(self.doc)
        
        db = Chroma.from_documents(self.documents,
                                   embedding=self.emb_model,
                                   persist_directory=self.vector_store_path)
        
        db.persist()
    
    def load_vector_db(self)-> Chroma:
        #to load back the embeddings from disk 
        db = Chroma(persist_directory=self.vector_store_path,embedding_function=self.emb_model)
        return db
    
    def get_retriever(self) -> Chroma:
        return self.load_vector_db().as_retriever()


