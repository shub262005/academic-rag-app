"""
app.py
------
Streamlit frontend for the Academic Document Assistant.
Provides:
- Sidebar with document list, Ollama status, and Rebuild Index button
- Suggested questions preview
- Chatbot interface with session history
- De-duplicated source citations for every answer
- Clean error messaging (no raw tracebacks)
"""

import os
import streamlit as st
import ollama_client
import rag

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academic Document Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished Academic UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    /* Status Badges */
    .status-badge-ok {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #DCFCE7;
        color: #166534;
        margin-bottom: 8px;
    }
    .status-badge-err {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #FEE2E2;
        color: #991B1B;
        margin-bottom: 8px;
    }
    /* Source box */
    .source-container {
        margin-top: 12px;
        padding: 10px 14px;
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .source-title {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 4px;
    }
    .source-item {
        color: #475569;
        margin-left: 12px;
    }
    /* Suggested questions card */
    .sample-box {
        background: #F1F5F9;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_checked" not in st.session_state:
    st.session_state.index_checked = False


# ---------------------------------------------------------
# Sidebar: Status, Documents & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎓 Academic Assistant")
    st.caption("College Document Q&A System")
    st.divider()

    # Ollama Health Check
    st.subheader("System Status")
    ollama_info = ollama_client.check_ollama_status()

    if ollama_info["connected"]:
        st.markdown('<span class="status-badge-ok">🟢 Ollama: Connected</span>', unsafe_allow_html=True)
        if ollama_info["chat_model_ready"] and ollama_info["embed_model_ready"]:
            st.caption("Models: `llama3.2:3b` & `nomic-embed-text` ready")
        else:
            st.warning("One or more required models not found. Pull `llama3.2:3b` and `nomic-embed-text`.")
    else:
        st.markdown('<span class="status-badge-err">🔴 Ollama: Not Connected</span>', unsafe_allow_html=True)
        st.error("Ollama is not running. Start Ollama and try again.")

    st.divider()

    # Documents Loaded
    st.subheader("Source Documents")
    docs = [
        ("Academic Calendar for Odd Sem 26-27.pdf", "📄", "Academic Calendar"),
        ("syllabus.pdf", "📄", "Syllabus"),
        ("time table.png", "🖼️", "Timetable (OCR)")
    ]

    for filename, icon, label in docs:
        exists = os.path.exists(filename)
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.write(icon)
        with col2:
            if exists:
                st.write(f"**{filename}**  \n<small style='color: #64748B;'>{label}</small>", unsafe_allow_html=True)
            else:
                st.write(f"**{filename}**  \n<small style='color: #EF4444;'>Missing file</small>", unsafe_allow_html=True)

    st.divider()

    # Index Management
    st.subheader("Index Management")
    index_ready = rag.is_index_ready()

    if index_ready:
        st.success("Vector Index: Loaded (ChromaDB)")
    else:
        st.info("Vector Index: Not built yet")

    if st.button("🔄 Rebuild Document Index", use_container_width=True):
        if not ollama_info["connected"]:
            st.error("Cannot rebuild index: Ollama is not running.")
        else:
            progress_placeholder = st.empty()
            with st.spinner("Processing source documents and updating ChromaDB..."):
                try:
                    count, msg = rag.build_vector_index(
                        workspace_dir=".",
                        force_rebuild=True,
                        progress_callback=lambda s: progress_placeholder.info(s)
                    )
                    progress_placeholder.empty()
                    st.success(f"Success! {count} chunks indexed.")
                    st.rerun()
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"Error rebuilding index: {e}")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------
# Automatic Initial Indexing on App Launch
# ---------------------------------------------------------
if not st.session_state.index_checked:
    st.session_state.index_checked = True
    if not rag.is_index_ready():
        if ollama_info["connected"]:
            with st.spinner("Initializing first-time document index... Please wait."):
                try:
                    count, _ = rag.build_vector_index(workspace_dir=".", force_rebuild=False)
                    st.toast(f"Indexed {count} chunks into ChromaDB!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to initialize document index: {e}")
        else:
            st.warning("Ollama is not running. Start Ollama and try again to build document index.")


# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
st.markdown('<div class="main-header">Academic Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Ask questions about your syllabus, timetable and academic calendar.</div>',
    unsafe_allow_html=True
)

# Suggested Questions Card (shown as reference)
with st.expander("💡 **Suggested Questions** (Click to view ideas)", expanded=(len(st.session_state.messages) == 0)):
    st.markdown("""
    You can ask questions like:
    - **What is my Monday timetable?**
    - **What is my first lecture on Tuesday?**
    - **What do I have after lunch on Wednesday?**
    - **Which days have Business Intelligence?**
    - **What topics are covered in Unit III of Business Intelligence?**
    - **What are the important dates in the academic calendar?**
    - **When does the semester start?**
    - **When are holidays scheduled?**
    - **What subjects are listed in my syllabus?**
    """)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            sources_html = "<div class='source-container'><div class='source-title'>Sources:</div>"
            for src in message["sources"]:
                sources_html += f"<div class='source-item'>• {src}</div>"
            sources_html += "</div>"
            st.markdown(sources_html, unsafe_allow_html=True)

# User Chat Input
user_query = st.chat_input("Ask a question about your academic documents...")

if user_query:
    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("Please enter a valid question.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": cleaned_query})
        with st.chat_message("user"):
            st.markdown(cleaned_query)

        # Check Ollama connection before invoking RAG
        status = ollama_client.check_ollama_status()
        if not status["connected"]:
            error_msg = "Ollama is not running. Start Ollama and try again."
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": []
            })
        elif not rag.is_index_ready():
            not_ready_msg = "The document index is not ready yet. Please click 'Rebuild Document Index' in the sidebar."
            with st.chat_message("assistant"):
                st.warning(not_ready_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": not_ready_msg,
                "sources": []
            })
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching academic documents and generating answer..."):
                    try:
                        result = rag.ask_assistant(cleaned_query, top_k=4)
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        if sources:
                            sources_html = "<div class='source-container'><div class='source-title'>Sources:</div>"
                            for src in sources:
                                sources_html += f"<div class='source-item'>• {src}</div>"
                            sources_html += "</div>"
                            st.markdown(sources_html, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    except ConnectionError:
                        err = "Ollama is not running. Start Ollama and try again."
                        st.error(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err,
                            "sources": []
                        })
                    except Exception as e:
                        err = f"An error occurred while answering your question: {e}"
                        st.error(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err,
                            "sources": []
                        })
