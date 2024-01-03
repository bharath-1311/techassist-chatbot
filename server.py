from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
from google.cloud.storage import Client
from PyPDF2 import PdfReader
import websockets
import asyncio
import sys
import json 
import io


with open('config.json') as f:
    config = json.load(f)
with open('static_qa.json') as g:
    static_qa = json.load(g)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['GOOGLE_APPLICATION_CREDENTIALS']
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

def replace_words(text, replacements):
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    return text


#os.environ['HUGGINGFACEHUB_API_TOKEN'] = config["HUGGINGFACEHUB_API_TOKEN"]
# repo_id = config["repo_id"]
# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 5000})



client = Client()
bucket = client.get_bucket(config['bucket'])
blob = bucket.get_blob(config['blob'])
content = blob.download_as_string()
pdf_stream = io.BytesIO(content)
pdfreader = PdfReader(pdf_stream)

raw_text ='''You are a helpful chatbot for helping technicians overcome issues. Assist me with all the numbered steps based the info provided. when asked, tell the complete details as follows
The jobs for the technician for today are:
1. Install order for Jacob Buchheim: Contact: 224-234-4781 Job Type: Install Host ID: W777395 Circuit ID: W777395.
2. Repair order for Jude Bellingham: Contact: 258-765-2497 Job Type: MMRC Host ID: W707953 Circuit ID: W707953.'''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

replacements = {"quantum": "xyz", "Quantum": "xyz", "lumen": "abc", "Lumen": "abc"}

raw_text = replace_words(raw_text, replacements)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)
#embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_texts(texts, embeddings)

qa_chain = load_qa_chain(llm=OpenAI(), chain_type="stuff", )
chat_history = []


async def chatbot(websocket, path):
    async for message in websocket:
        query = message.strip()
        query = replace_words(query, replacements)
        static_qa_lower = {key.lower(): value for key, value in static_qa.items()}
        query_lower = query.lower().strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting")
            await websocket.send("Exiting")
            break
        elif query_lower in static_qa_lower:
            result = f'{static_qa_lower[query_lower]}'
            accumulated_response = ""
            if isinstance(result, dict):
                for key, value in result.items():
                    accumulated_response += f"{key}: {value}\n"
            else:
                accumulated_response = result
            await websocket.send(accumulated_response)            
        else:
            context = "".join(chat_history[-3:])
            docs = faiss_index.similarity_search(query)
            result = qa_chain.run(
                input_documents=docs,
                question=query,
                context=context,
                #prompt="Im a technician. Give all steps related to my issue/query . my query: {query}",
            )
            chat_history.append(result)
            await websocket.send(result)


if __name__ == "__main__":
    start_server = websockets.serve(chatbot, "localhost", 8765)
    print("WebSocket server started. Listening on localhost:8765")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
