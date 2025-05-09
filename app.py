import streamlit as st
st.set_page_config(page_title="RAG Chatbot", layout="wide")
import pandas as pd
import numpy as np

# Embedding Finetune Zip
import tempfile
import zipfile
import os 

with st.sidebar:
    uploaded_model = st.file_uploader("Upload Fine-tuned Embeddings (ZIP)", type="zip")
    
    if uploaded_model:
        # Create persistent temp dir
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(uploaded_model) as z:
                z.extractall(temp_dir)

            # Detect inner folder if it exists
            subdirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
            if subdirs:
                model_path = subdirs[0]  # Use inner model folder
            else:
                model_path = temp_dir  # Use root if no subfolder

            st.success("Embeddings loaded!")

        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

# Open AI Key
import openai
[openai]
OPEN_API_KEY = "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"
# os.environ["OPENAI_API_KEY"] = "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"
from langchain_openai import ChatOpenAI

# Chunking and Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embedding Finetuning Imports
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Rag Pipeline Imports
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate

#Import Scrapped Data
data=pd.read_csv("rag_inputs.csv",index_col=0)

#Initialize Model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0) # gpt

#Streamlit Title
# st.title("RAG Chatbot: MS in Applied Data Science")
# user_input = st.text_input("Ask a question:")

st.markdown("""
    <style>
    .stChatMessage.user {background-color: #DCF8C6; text-align: right;}
    .stChatMessage.assistant {background-color: #F1F0F0; text-align: left;}
    .stTextInput>div>div>input {
        background-color: #fff;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F916 RAG Chatbot for MS in Applied Data Science")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#Store Embedding and Vectorstore
@st.cache_resource
# def prepare_vectorstore(data):

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)
#     all_chunks = []
#     for idx, row in data.iterrows():
#         combined_text = row['Title'] + "\n" + row['Text']
#         chunks = splitter.split_text(combined_text)
#         for chunk in chunks:
#             all_chunks.append(Document(
#                 page_content=chunk,
#                 metadata={"title": row["Title"], "url": row["URL"]}
#             ))
#     embedding = HuggingFaceEmbeddings(model_name="exp_finetune")
#     return FAISS.from_documents(all_chunks, embedding=embedding)

# vectorstore = prepare_vectorstore(data)

def prepare_vectorstore(data, embedding_model_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)
    all_chunks = []
    for idx, row in data.iterrows():
        combined_text = row['Title'] + "\n" + row['Text']
        chunks = splitter.split_text(combined_text)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata={"title": row["Title"], "url": row["URL"]}
            ))
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_path)
    return FAISS.from_documents(all_chunks, embedding=embedding)

## Initialize vectorstore ONLY after upload ##
if uploaded_model and "vectorstore" not in st.session_state:
    st.session_state.vectorstore = prepare_vectorstore(data, model_path)

vectorstore = st.session_state.get("vectorstore", None)


def decompose_query(question: str) -> list:
    prompt = f"""Decompose the following question into 2–4 smaller sub-questions:\n\nQuestion: {question}\n\nSub-questions:"""
    result = llm.invoke(prompt)
    text = result.content  # 👈 extract the string from AIMessage
    subquestions = text.strip().split("\n")
    return [sq.strip("-• ") for sq in subquestions if sq.strip()]

def retrieve_all_subq_docs(question: str):
    subqs = decompose_query(question)
    docs = []
    for sq in subqs: # for queries from above, run search to get top k, and append to docs - this could have duplicated chunks
        docs.extend(vectorstore.similarity_search(sq, k=3))
    return docs

def format_docs(docs):
    return "\n\n".join(f"[{doc.metadata['title']}] {doc.page_content}" for doc in docs)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant helping answer questions about the MS in Applied Data Science program.
Use the context below to answer the question at the end. (3 - 5 sentences)
If you don't know the answer, just return \"I'm not sure\" and do not invent facts.

Context:
{context}

Question:
{question}

Answer in a detailed, professional way:
""",
)

# 7. Build the chain
rag_chain = (
    {"context": RunnableLambda(retrieve_all_subq_docs) | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Run chain and show result
# if user_input:
#     with st.spinner("Thinking..."):
#         answer = rag_chain.invoke(user_input)
#         st.markdown("### Answer")
#         st.write(answer)

# 8. Chat Interface
user_input = st.chat_input("Ask a question about the program...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        answer = rag_chain.invoke(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])