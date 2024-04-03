from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st
import io
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community import embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import csv
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import pandas as pd

llm = Ollama(model="llama2")


load_dotenv(find_dotenv(), override=True)

title_container = st.container()
with title_container:
    st.title("AI Data Analytics Application on CSV data")

uploaded_file = st.sidebar.file_uploader(
    "Upload File For Data Analysis",
    type="csv",
    accept_multiple_files=False,
    key="file-uploader-side-bar",
)

if uploaded_file:
    # try:
    columns_to_embed = ["Number", " Incident Description", " Notes"]
    columns_to_metadata = [
        "Number",
        " Incident State",
        " Active",
        " Reassignment Count",
        " Reopen Count",
        " Sys Mod Count",
        " Made SLA",
        " Caller ID",
        " Opened By",
        " Opened At",
        " Sys Created By",
        " Sys Created At",
        " Sys Updated By",
        " Sys Updated At",
        " Contact Type",
        " Location",
        " Category",
        " Subcategory",
        " U Symptom",
        " CMDB CI",
        " Impact",
        " Urgency",
        " Priority",
        " Assignment Group",
        " Assigned To",
        " Knowledge",
        " U Priority Confirmation",
        " Notify",
        " Problem ID",
        " RFC",
        " Vendor",
        " Caused By",
        " Closed Code",
        " Resolved By",
        " Resolved At",
    ]

    docs = []
    df = pd.read_csv(uploaded_file,sep =";")
    for index, row in df.iterrows():
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(
            f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items()
        )
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=0, length_function=len
    )

    all_splits = splitter.split_documents(docs)
    embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)
    # except Exception as e:
    #     st.error(f"Error loading CSV file: {str(e)}")
    #     st.stop()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def chat_actions():
    st.session_state["chat_history"].append(
        HumanMessage(content=st.session_state["Chat_input-chatbot"]),
        # {"role": "user", "content": st.session_state["Chat_input-chatbot"]},
    )
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                    which can be understood without the chat history. Do NOT answer the question, \
                        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        qa_system_prompt = """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                If you don't know the answer, just say that you don't know. \
                Use three sentences maximum and keep the answer concise.\
                {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        inputQ = st.session_state["Chat_input-chatbot"]
        output_result = rag_chain.invoke(
            {"input": inputQ, "chat_history": st.session_state["chat_history"]}
        )
        st.session_state["chat_history"].append(
            AIMessage(content=output_result["answer"])
        )

        for i in st.session_state["chat_history"]:
            with st.chat_message(name=i.dict()["type"]):
                st.write(i.content)

    except ValueError as e:
        if "special characters" in str(e):
            st.error(
                f"Prompt contains special characters not allowed by TikToken."
                f"Please remove those characters or try adjusting TikToken configuration."
            )
        else:
            raise
    except NameError:
        with st.chat_message(name="assistant"):
            st.write("Input A file to use Data Analysis AI")


prompt = st.chat_input(
    "Say something", key="Chat_input-chatbot", on_submit=chat_actions
)
