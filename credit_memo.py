import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI
from gtts import gTTS
import base64
import streamlit.components.v1 as components
import asyncio
import pandas as pd

# Load environment variables
load_dotenv()

PERSIST_DIR = "./db"
DATA_DIR = "data"
CHUNK_SIZE = 100  # Number of pages to process at a time

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

async def data_ingestion():
    pdf_data = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DATA_DIR, filename)
            loader = PyMuPDFLoader(filepath)
            doc_pages = loader.load()

            # Process in chunks
            documents = []
            for i in range(0, len(doc_pages), CHUNK_SIZE):
                chunk = doc_pages[i:i + CHUNK_SIZE]
                documents.extend(chunk)

            # Initialize HuggingFace embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

            # Create a FAISS vector store from the documents
            vector_store = FAISS.from_documents(documents, embeddings)

            # Save the vector store to disk
            vector_store.save_local(os.path.join(PERSIST_DIR, filename))
            pdf_data[filename] = vector_store

    st.session_state.pdf_data = pdf_data
    st.success("Data ingestion completed successfully!")

async def handle_query(query):
    try:
        responses = []
        for filename, vector_store in st.session_state.pdf_data.items():
            # Load the vector store with dangerous deserialization allowed
            vector_store = FAISS.load_local(
                os.path.join(PERSIST_DIR, filename),
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
            )

            # Retrieve relevant documents
            docs = vector_store.similarity_search(query)
            #st.write(docs)

            # Combine the content of the retrieved documents
            context = "\n".join([doc.page_content for doc in docs])

            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""You are a Q&A assistant named Chat-bot, created by Pratap. You have a specific response programmed for when users specifically ask about your creator, Pratap. The response is: "I was created by Pratap, an enthusiast in Artificial Intelligence." For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
                Context:
                {context}
                Question:
                {query}
                """
            )

            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            chain = LLMChain(
                llm=llm,
                prompt=prompt_template,
                callbacks=[StreamlitCallbackHandler(st.container())]
            )
            
            response = await chain.arun(context=context, query=query)
            responses.append(response)

        final_response = "\n\n".join(responses)
        return final_response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

async def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        audio_file = open("output.mp3", "rb")
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio id="audio">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <button id="play-button">Play Audio</button>
        <script>
            var audio = document.getElementById('audio');
            var playButton = document.getElementById('play-button');
            var isPlaying = false;

            playButton.addEventListener('click', function() {{
                if (isPlaying) {{
                    audio.pause();
                    playButton.innerText = 'Play Audio';
                }} else {{
                    audio.play();
                    playButton.innerText = 'Pause Audio';
                }}
                isPlaying = !isPlaying;
            }});
        </script>
        """
        components.html(audio_html, height=100)
    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {e}")

st.title("EXL Chatbot🤖")
st.markdown("Created by Pratap") 

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Please upload a PDF and ask a question.'}]
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = {}

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    if uploaded_files and st.button("Submit"):
        with st.spinner("Processing..."):
            for uploaded_file in uploaded_files:
                filepath = os.path.join(DATA_DIR, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            asyncio.run(data_ingestion())  
            st.success("Done")

user_prompt = st.chat_input("Ask me")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = asyncio.run(handle_query(user_prompt))
    st.session_state.messages.append({'role': 'assistant', "content": response})
    asyncio.run(text_to_speech(response))

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Add the play button near the chat bar
st.markdown("""
    <style>
        .stChatInput {
            display: flex;
            align-items: center;
        }
        .stChatInput button {
            margin-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)