import os
import glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load ENV only once
# -----------------------------
@st.cache_resource
def load_config():
    load_dotenv()
    return {
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.getenv("API_KEY"),
        "model": os.getenv("MODEL_DEPLOYMENT")
    }

# -----------------------------
# Initialize client (cached)
# -----------------------------
@st.cache_resource
def init_client():
    config = load_config()
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["endpoint"]
    )
    return client

# -----------------------------
# Load PDFs (only 1–2)
# -----------------------------
@st.cache_resource
def load_files():
    files = glob.glob("brochures/*.pdf")[:2]  # limit to 2 files
    return files

# -----------------------------
# Create vector store (cached)
# -----------------------------
@st.cache_resource
def create_vector_store(client, files):
    print("Creating vector store... (only once)")

    file_streams = [open(file, "rb") for file in files]

    vector_store = client.vector_stores.create(name="travel-brochures")

    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams
    )

    return vector_store

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("🌍 Travel Assistant (Fast Version)")

    client = init_client()
    files = load_files()
    vector_store = create_vector_store(client, files)

    user_input = st.text_input("Ask a question about travel brochures:")

    if user_input:
        with st.spinner("Thinking..."):

            response = client.responses.create(
                model=load_config()["model"],
                input=user_input,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id]
                }]
            )

            st.write(response.output_text)

# -----------------------------
if __name__ == "__main__":
    main()
