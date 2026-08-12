import streamlit as st
import os
from src.ingestion import ingest_document, ingest_raw_text
from src.query_engine import get_answer

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
            # if new file
            if uploaded_file.name != st.session_state.get("last_uploaded_file"):
                # temporarily save the uploaded file to disk.
                temp_file_path = f"./data/raw/temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Processing and indexing document..."):
                    ingest_document(temp_file_path)
                
                os.remove(temp_file_path)
                st.session_state.last_uploaded_file = uploaded_file.name
                st.success("Document ingested successfully! Database updated.")
        

    # TAB 2: raw text pasting logic
    with tab2:
        # Clear the text area before instantiation, if flagged
        if st.session_state.get("clear_pasted_text", False):
            st.session_state.pasted_text = ""
            st.session_state.clear_pasted_text = False

        pasted_text = st.text_area(
            "Paste your raw text here:",
            height=200,
            key="pasted_text"
        )

        if st.button("Ingest Text"):
            if pasted_text.strip():
                with st.spinner("Processing pasted text..."):
                    ingest_raw_text(pasted_text)

                st.session_state.clear_pasted_text = True
                st.session_state.show_ingest_success = True
                st.rerun()
            else:
                st.warning("Please paste some text first.")

         # Show pending success message from the previous run, then clear it
        if st.session_state.get("show_ingest_success", False):
            st.success("Text ingested successfully!")
            st.session_state.show_ingest_success = False
  
    
# --- MAIN AREA: Query Pipeline ---
st.header("2. Ask a Question")
user_query = st.text_input("What are you looking for?")

# "Ask" button click
if st.button("Ask"):
    if user_query:
        with st.spinner("Searching documents and thinking..."):
            # Call Pipeline B
            answer, sources = get_answer(user_query)
            
        # final answer
        st.write("### Answer:")
        st.write(answer)
        
        # collapsible expander to show the source chunks (Metadata/Citation!)
        with st.expander("View Source Chunks"):
            for i, chunk in enumerate(sources, start=1):
                st.write(f"**Chunk {i}:**")
                st.write(chunk.page_content)
                st.write("---")
                
    else:
        st.warning("Please type a question first.")