import pickle

from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.prompts import PromptTemplate
from pathlib import Path
import os
import openai

load_dotenv()
import streamlit as st
OPENAI_KEY = os.getenv('OPENAI_KEY')
# https://pypi.org/project/youtube-transcript-api/
st.title('Youtuber AI Chatbot ✨')
st.subheader("Follow [@jamescodez](https://twitter.com/jamescodez) on twitter for more!")

_template = """ Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI version of the youtuber {name} .
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
Question: {question}
=========
{context}
=========
Answer:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context", "name"])

# video_ids = ["ReeLQR7KCcM", "D4YTJ2W5q4Y", "Qnos8aApa9g", "hW5s_UUO1RI"]
script_docs = []

# You can change this so a user can input multiple video IDs
# And generate the embeddings from multiple videos, but more expensive
video1 = st.text_input(label="Enter Youtube Video ID")
youtuberName = st.text_input(label="Name", placeholder="youtuber name")
# video2 = st.text_input(label="Enter Youtube Video Link 2")
if video1 != "" and youtuberName != "":
    t = YouTubeTranscriptApi.get_transcript(video1)
    finalString = ""
    for item in t:
        text = item['text']
        finalString += text + " "
    print(finalString)
    # load data sources to text (yt->text)
    text_splitter = CharacterTextSplitter()
    chunks = text_splitter.split_text(finalString)
    vectorStorePkl = Path("vectorstore.pkl")
    vectorStore = None
    # if vectorStorePkl.is_file():
    #     print("vector index found.. ")
    #     with open('vectorstore.pkl', 'rb') as f:
    #         vectorStore = pickle.load(f)
    # else:
    print("regenerating search index vector store..")
    # It uses OpenAI API to create embeddings (i.e. a feature vector)
    # https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
    vectorStore = FAISS.from_texts(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_KEY))
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorStore, f)

    qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_KEY),
                                    vectorstore=vectorStore, qa_prompt=QA_PROMPT)

    chat_history = []
    userInput = st.text_input(label="Question", placeholder="Ask the AI youtuber a question!")
    if userInput != "":
        result = qa({"name": youtuberName, "question": userInput, "chat_history": chat_history}, return_only_outputs=True)
        print(result)
        st.subheader(result["answer"])
    # create embeddings

# load embeddings to vector store

