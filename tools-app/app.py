import os
import glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- LOAD ENV --------------------
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("API_KEY")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT")

# -------------------- UI SETUP --------------------
st.set_page_config(page_title="Margie's Travel Assistant", page_icon="✈️")
st.title("✈️ Margie's Travel Assistant")
st.caption("Ask about destinations, hotels, and travel services!")

# -------------------- CLIENT INIT --------------------
openai_client = OpenAI(
    api_key=os.getenv("API_KEY")
   
)

# -------------------- VECTOR STORE SETUP --------------------
if "vector_store_id" not in st.session_state:
    with st.spinner("📂 Loading travel brochures..."):

        try:
            # Create vector store
            vector_store = openai_client.vector_stores.create(
                name="travel-brochures"
            )

            # Load PDFs
            pdf_files = glob.glob("brochures/*.pdf")

            if not pdf_files:
                st.error("❌ No PDF files found in /brochures folder")
                st.stop()

            file_streams = [open(file, "rb") for file in pdf_files]

            try:
                file_batch = openai_client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id,
                    files=file_streams
                )
            finally:
                for f in file_streams:
                    f.close()

            st.session_state.vector_store_id = vector_store.id

            st.success(f"✅ Loaded {file_batch.file_counts.completed} brochures")

        except Exception as e:
            st.error("❌ Failed to load brochures")
            st.exception(e)
            st.stop()

# -------------------- CHAT STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- USER INPUT --------------------
if prompt := st.chat_input("Ask about travel destinations or hotels..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):

            try:
                response = openai_client.responses.create(
                    model=MODEL_DEPLOYMENT,  # MUST be Azure deployment name
                    instructions="""
You are a travel assistant for Margie's Travel.

- Use the uploaded brochures for company-specific answers.
- Use web search for general travel info.
- Give clear, helpful, structured answers.
""",
                    input=prompt,
                    previous_response_id=st.session_state.last_response_id,
                    tools=[
                        {
                            "type": "file_search",
                            "vector_store_ids": [st.session_state.vector_store_id],
                        },
                        {
                            "type": "web_search_preview"
                        }
                    ]
                )

                answer = response.output_text

                st.markdown(answer)

                # Save state
                st.session_state.last_response_id = response.id
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error("❌ Error generating response")
                st.exception(e)
