import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
prompt_template= """

Human: Use the following pieces of context to provide
a concise answer. If you dont know the answer then do 
not answer if you have no reference 
<<context>>
{context}
</context

Question: {question}
Assistant: ""

"""
import os 
from dotenv import load_dotenv

load_dotenv()
aws_access_key_id=os.getenv('aws_access_key_id')

aws_secret_access_key_id=os.getenv('aws_secret_access_key')

bedrock_client=boto3.client(
    service_name="bedrock-runtime",region_name='us-east-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key_id

)
#get embedding model from bedrock 
embedding=BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=bedrock_client)

def get_documents():
    loader = PyPDFDirectoryLoader("docs")
    documents=loader.load()
    splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500)
    docs=splitter.split_documents(documents)
    return docs

def get_vector_store(docs):

    vectorstore_faiss=FAISS.from_documents(
        docs, embedding
    )
    vectorstore_faiss.save_local('faiss_local')


def get_llm():
    llm=Bedrock(model_id="mistral.mistral-7b-instruct-v0:2",
    client=bedrock_client)
    return llm

PROMPT=PromptTemplate(
    template=prompt_template,
    input_variables=['context','question']
)

def get_llm_response(llm, vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff', #stuff for qa chain type 
        #refer documentation
        retriever=vectorstore_faiss.as_retriever(
            search_kwargs={'k':3},
            return_source_documents=True,
            chain_type_kwargs={"prompt":PROMPT}
        )
    )
    response=qa({'query':query})
    return response['result']

def main():
    st.set_page_config("RAG")
    st.header("End to End RAG AWS Bedrock")
    user_question=st.text_input("Ask Question from the PDF file")
    
    with st.sidebar:
        st.title('Update & create vectorstore')

        if st.button('Store Vector'):
            with st.spinner("Processing..."):
                docs=get_documents()
                get_vector_store(docs)
                st.success("Done")
        if st.button('Send'):
            with st.spinner("Processing..."):
                faiss_index=FAISS.load_local('faiss_local', embedding, allow_dangerous_deserialization=True)
                llm=get_llm()
                st.write(get_llm_response(llm,faiss_index,user_question))


        


  
if __name__ == "__main__":
    main()

