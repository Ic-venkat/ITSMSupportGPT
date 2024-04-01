import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from langchain_contrib.prompts import Chain, ConcatenationPromptTemplate
from langchain.llms.embeddings import OpenAIEncoder
from langchain.storage import PineconeClient
from langchain.agents import RetrievalAgent
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv(), override=True)

# Streamlit app title
st.title("AI Data Analytics and Chat Application")

# File upload for data analysis
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File for Analysis",
    type="csv",
    accept_multiple_files=False,
    key="file-uploader-side-bar",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="ANSI")
    st.dataframe(df)

    # Descriptive summaries for each column (using Chain prompt)
    description_prompt = Chain(
        inputs=["column_name"],
        outputs=["description"],
        template=ConcatenationPromptTemplate(
            input_separator="\n",
            output_separator="\n",
            prefix="**Column Name:** {column_name}\n",
            suffix="\n**Description:** {description}",
        ),
    )

    def summarize_column(column_name):
        description = llm.run(description_prompt, column_name=column_name)
        return description["description"].strip()

    df_descriptions = df.columns.apply(summarize_column).to_frame("Description")
    st.write("**Descriptive Summaries:**")
    st.dataframe(df_descriptions)

    # OpenAI API key (replace with your actual key)

    # Initialize OpenAI LLM and encoder
    llm = OpenAI()
    encoder = OpenAIEncoder(llm=llm)

    # Pinecone database configuration (replace with your credentials)
    from pinecone import PodSpec

    if "langchain" not in pc.list_indexes().names():
        print(f"Creating Index langchain")
        pc.create_index(
            name="langchain",
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        print("Index Created")
    else:
        print(f"Index langcain already exist")
        
    index = pc.Index("langchain")

    # Initialize Pinecone client
    pinecone_client = PineconeClient(
        api_key=pinecone_api_key, environment=pinecone_environment
    )

    # Vector embeddings using OpenAI encoder
    embeddings = encoder.encode_texts(df.values.flatten())

    # Save embeddings to Pinecone database
    index.upsert(embeddings, ids=df.index.tolist())

    # Create RAG retrieval agent
    rag_agent = RetrievalAgent(
        pinecone_client=pinecone_client,
        encoder=encoder,
        index_name="pinecone",  # Replace with your Pinecone index name
        reader=df,  # Use DataFrame as the knowledge base
    )

    # Chat interaction with RAG model
    prompt = st.chat_input(
        "Ask a question about the data",
        key="Chat_input-chatbot",
        on_submit=chat_actions,
    )

    def chat_actions():
        if prompt is not None:
            response = rag_agent.run(prompt)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(name=message["role"]):
            st.write(message["content"])
