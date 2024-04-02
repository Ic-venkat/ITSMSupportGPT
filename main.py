from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import streamlit as st
import io

load_dotenv(find_dotenv(), override=True)

title_container = st.container()
with title_container:
    st.title("AI Data Analytics Application on CSV data")

uploaded_file = st.sidebar.file_uploader("Upload File For Data Analysis", type="csv", accept_multiple_files= False, key="file-uploader-side-bar")

if(uploaded_file):
    try:
        loader = CSVLoader(uploaded_file.name)
        index_creator = VectorstoreIndexCreator()
        docsearch = index_creator.from_loaders([loader])
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.stop()

if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
        
def chat_actions():
    st.session_state["chat_history"].append(
        {"role": "user", "content": st.session_state["Chat_input-chatbot"]},
    )
    try:
        
        prompt = st.session_state["Chat_input-chatbot"]
        chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=docsearch.vectorstore.as_retriever(k=10),
            input_key="question",
        )
        output_result = chain({"question": prompt})['result']
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": output_result}
        )
        for i in st.session_state["chat_history"]:
            with st.chat_message(name=i["role"]):
                st.write(i["content"])
        
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




prompt = st.chat_input("Say something",key='Chat_input-chatbot',on_submit = chat_actions)







