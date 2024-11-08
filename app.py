from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st
import os
import pandas as pd
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from prompt import prompts
from langfuse import Langfuse
from langfuse.decorators import langfuse_context
import guardrails as gr
import json


load_dotenv()

pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index("moviesuggestions")

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_MODEL_DEPLOYMENT"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),

)

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("LLM_ENDPOINT"),
    azure_deployment=os.getenv("LLM_DEPLOYMENT"),
    openai_api_version="2024-05-01-preview",
    api_key=os.getenv("LLM_API_KEY"),
    model="gpt-4o-mini",

)

langfuse_client = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST"),
)

with open("movie_guardrail.json") as file:
    guardrail_config = json.load(file)

docSearch = Pinecone(index=index, embedding=embedding_model, text_key="title")
retriever = docSearch.as_retriever()

def read_data_from_csv():
    print("Reading data from CSV file...")
    df = pd.read_csv("tmdb_5000_credits.csv")
    for idx, row in df.iterrows():
        text = row['title']
        home_page = row['homepage']
        embedding = embedding_model.embed_query(text)
        language = row['original_language']
        plot = row['overview']
        embedding = embedding_model.embed_query(plot)
        pinecone_id = f"record-{idx}"
        index.upsert([(pinecone_id, embedding, {"title": text, "genre": row['genres'], "language": language, "home_page": home_page, "poster": row['poster']})])
    print("Data read and stored in Pinecone successfully.")
st.title("Movie Suggestion App")

left, middle, right = st.columns(3)
with left:
    st.text("Movie Suggestion App")
with right:
    st.button("Load Movies", on_click=read_data_from_csv, use_container_width=False)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_movie_suggestion(rag_chain, user_message):
    print("Getting movie suggestion for genre:")
    movie_suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["movie_suggestion"]),
            ("human", "{input}")
        ])
    movie_suggestion_answer_chain = create_stuff_documents_chain(llm, movie_suggestion_prompt)
    rag_chain = create_retrieval_chain(retriever, movie_suggestion_answer_chain)
    return rag_chain.invoke({"input": f"Use the following {user_message} and return the response"})

def get_movie_name_from_story(rag_chain, user_message):
    print("getting movie name from story")
    movie_suggestion_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["movie_name"]),
            ("human", "{input}")
        ])
    movie_suggestion_answer_chain = create_stuff_documents_chain(llm, movie_suggestion_prompt)
    rag_chain = create_retrieval_chain(retriever, movie_suggestion_answer_chain)
    return rag_chain.invoke({"input": f"Use the following {user_message} and return the response"})


def handle_message():
    user_message = st.session_state.user_input
    if user_message:
        st.session_state.chat_history.append({"sender": "user", "text": user_message})

        movie_qa_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["retrive_from_message"]),
            ("human", "{input}")
        ])
        movie_suggestion_answer_chain = create_stuff_documents_chain(llm, movie_qa_prompt)
        rag_chain = create_retrieval_chain(retriever, movie_suggestion_answer_chain)
        out = rag_chain.invoke({"input": user_message})
        print(out)
        match out['answer']:
            case "movie_suggestion":
                print("inside movie suggestion")
                response = get_movie_suggestion(rag_chain, user_message)
            case "movie_name":
                print("inside movie name")
                response = get_movie_name_from_story(rag_chain, user_message)
            case "not_clear":
                response = {"answer": "I'm sorry, I didn't understand your request. Please try again."}

        langfuse_client.generation(
            trace_id=langfuse_context.get_current_trace_id(),
            parent_observation_id=langfuse_context.get_current_observation_id(),
            model="gpt-4o-mini",
            metadata={
                "prompt": prompts["retrive_from_message"],
                "input": user_message,
                "output": response['answer']
            },
            name="Movie Suggestion",
            input=user_message,
            output=response['answer'],
        )
    if response:
        st.session_state.chat_history.append({"sender": "bot", "text": response['answer']})
        print(response['answer'])
        st.session_state.user_input = ""    

for msg in st.session_state.chat_history:
    if msg["sender"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("bot").write(msg["text"])

st.text_input(
    label="",
    placeholder="Type your message",
    key="user_input",
    on_change=handle_message
)
