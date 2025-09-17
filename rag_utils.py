from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

import config
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

_embedding_instance = None
_model_instance = None
_tokenizer = None

def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is None:
        if config.local_emb_path is None:
            raise ValueError("⚠️ config.local_emb_path ainda não foi inicializado!")
        _embedding_instance = HuggingFaceEmbeddings(model_name=config.local_emb_path, model_kwargs={"device": "cpu"})
    return _embedding_instance

def get_model():
    global _model_instance
    if _model_instance is None:
        if config.local_model_path is None:
            raise ValueError("⚠️ config.local_model_path ainda não foi inicializado!")
        _model_instance = AutoModelForCausalLM.from_pretrained(
            config.local_model_path,
            dtype=torch.float16,
            device_map={"": "cuda"},
            trust_remote_code=True
        )
    
    return _model_instance

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        if config.local_model_path is None:
            raise ValueError("⚠️ config.local_model_path ainda não foi inicializado!")
        _tokenizer = AutoTokenizer.from_pretrained(config.local_model_path, trust_remote_code=True)
    
    return _tokenizer

def create_split_doc(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([raw_text])

    return docs

def store_docs(docs):
    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

def create_template():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um especialista em extrair informações em documentos.
Com base nas informações a seguir, forneça a melhor resposta.
Caso não tenha certeza da resposta, prefira falar que não sabe responder tal pergunta.
Responda de maneira amigável e clara.

Contexto:
{context}

Pergunta:
{question}
"""
)
    return prompt_template

def create_rag_chain(vectorstore):
    pipe = pipeline(
        "text-generation",
        model=get_model(),
        tokenizer=get_tokenizer(),
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        return_full_text=False
    )

    # Adapta para LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": create_template()}
    )
    
    return rag_chain

if __name__ == '__main__':
    pass