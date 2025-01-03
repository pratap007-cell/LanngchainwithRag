import os
import asyncio
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain_openai import AzureChatOpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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

            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(os.path.join(PERSIST_DIR, filename))
            pdf_data[filename] = vector_store

    st.session_state.pdf_data = pdf_data
    st.success("Data ingestion completed successfully!")

async def handle_query(query):
    try:
        responses = []
        for filename, vector_store in st.session_state.pdf_data.items():
            vector_store = FAISS.load_local(
                os.path.join(PERSIST_DIR, filename),
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
            )
            docs = vector_store.similarity_search(query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""Process the query and return content in a structured format:
                {context}
                Query: {query}"""
            )
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = await chain.arun(context=context, query=query)
            responses.append(response)
        return "\n\n".join(responses)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

def generate_pdf(data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Generated Document")
    y_position = 730
    for section, content in data.items():
        c.drawString(100, y_position, f"{section}: {content}")
        y_position -= 20
        if y_position < 50:  # Prevent writing beyond the page
            c.showPage()
            y_position = 750
    c.save()

st.title("PDF Processing Chatbot")
st.markdown("Upload PDFs and ask questions. Outputs are generated in structured PDF format.")

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

user_prompt = st.text_input("Ask me a question:")
if user_prompt:
    response = asyncio.run(handle_query(user_prompt))
    formatted_data = {"Response": response}
    generate_pdf(formatted_data, "output.pdf")
    displayPDF("output.pdf")
    st.success("Generated PDF is ready!")
