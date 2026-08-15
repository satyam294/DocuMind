import streamlit as st
import os
from src.ingestion import ingest_document, ingest_raw_text, get_available_documents
from src.query_engine import get_answer

if "available_docs" not in st.session_state:
    st.session_state.available_docs = get_available_documents()

st.title("DocuMind - Query your Docs")

# --- SIDEBAR: Data Ingestion ---
with st.sidebar:
    st.header("1. Upload Document")

    # two tabs for our ingestion methods
    tab1, tab2 = st.tabs(["Upload File", "Paste Text"])

    # TAB 1: file upload logic
    with tab1:
        uploaded_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"])
            
        if uploaded_file is not None:

            if st.button("Ingest File"):
    
                # temporarily save the uploaded file to disk.
                temp_file_path = f"./data/raw/temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Processing document..."):
                    is_new = ingest_document(temp_file_path, uploaded_file.name)

                if is_new:
                    st.success("Document ingested successfully!")
                    st.session_state.available_docs = get_available_documents()
                    st.rerun()
                else:
                    st.error("Duplicate document. Skipped ingestion.")
                
                os.remove(temp_file_path)
            
        

    # TAB 2: raw text pasting logic
    with tab2:
        # Clear the text area before instantiation, if flagged
        if st.session_state.get("clear_pasted_text", False):
            st.session_state.pasted_text = ""
            st.session_state.text_title = ""
            st.session_state.clear_pasted_text = False

        text_title = st.text_input("Give this text a title:")

        pasted_text = st.text_area(
            "Paste your raw text here:",
            height=200,
            key="pasted_text"
        )

        if st.button("Ingest Text"):
            if pasted_text.strip() and text_title.strip():
                with st.spinner("Processing pasted text..."):
                    is_new = ingest_raw_text(pasted_text, text_title)

                if is_new:
                    st.success("Text ingested successfully!")
                    st.session_state.clear_pasted_text = True
                    st.session_state.show_ingest_status = True
                    st.session_state.available_docs = get_available_documents()
                    st.rerun()
                else:
                    st.error("Duplicate document. Skipped ingestion.")

            else:
                st.warning("Please provide both a title and text.")

        # Show pending success message from the previous run, then clear it
        if st.session_state.get("show_ingest_status", False):
            st.success("Text ingested successfully!")
            st.session_state.show_ingest_status = False
  
    
# --- MAIN AREA: Query Pipeline ---
st.header("2. Ask a Question")
available_docs = st.session_state.available_docs

selected_docs = st.multiselect(
    "Filter by specific documents (leave empty to search all):",
    options=available_docs
)

user_query = st.text_input("What are you looking for?")

# "Ask" button click
if st.button("Ask"):
    if user_query:
        with st.spinner("Searching documents and thinking..."):
            # Call Pipeline B
            answer, sources = get_answer(user_query, selected_files=selected_docs)
            
        # final answer
        st.write("### Answer:")
        st.write(answer)
        
        # collapsible expander to show the source chunks (Metadata/Citation!)
        with st.expander("View Source Chunks"):
            for i, chunk in enumerate(sources, start=1):
                source_name = chunk.metadata.get('file_name', 'Unknown')
                st.write(f"**Chunk {i} (from {source_name}):**")
                st.write(chunk.page_content)
                st.write("---")
                
    else:
        st.warning("Please type a question first.")