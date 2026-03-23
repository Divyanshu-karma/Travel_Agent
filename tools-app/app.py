import os
import glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load env
load_dotenv()
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("API_KEY")
model_deployment = os.getenv("MODEL_DEPLOYMENT")

# Page config
st.set_page_config(page_title="Margie's Travel Assistant", page_icon="✈️")
st.title("✈️ Margie's Travel Assistant")
st.caption("Ask about destinations, hotels, and travel services!")

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=azure_openai_endpoint,
    api_key=api_key
)

# Create vector store only once using session state
if "vector_store_id" not in st.session_state:
    with st.spinner("Loading travel brochures..."):
        vector_store = openai_client.vector_stores.create(
            name="travel-brochures"
        )
        file_streams = [open(f, "rb") for f in glob.glob("brochures/*.pdf")]
        file_batch = openai_client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
        for f in file_streams:
            f.close()
        st.session_state.vector_store_id = vector_store.id
        st.success(f"Loaded {file_batch.file_counts.completed} brochures!")

# Track conversation
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about travel destinations or hotels..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            response = openai_client.responses.create(
                model=model_deployment,
                instructions="""
                You are a travel assistant that provides information on travel services available from Margie's Travel.
                Answer questions about services offered by Margie's Travel using the provided travel brochures.
                Search the web for general information about destinations or current travel advice.
                """,
                input=prompt,
                previous_response_id=st.session_state.last_response_id,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [st.session_state.vector_store_id]
                    },
                    {
                        "type": "web_search_preview"
                    }
                ]
            )
            answer = response.output_text
            st.markdown(answer)
            st.session_state.last_response_id = response.id
            st.session_state.messages.append({"role": "assistant", "content": answer})
