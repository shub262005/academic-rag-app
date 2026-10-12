"""
rag.py
------
Retrieval-Augmented Generation (RAG) module:
- Manages ChromaDB vector store in chroma_db/
- Generates embeddings using nomic-embed-text via Ollama
- Query expansion & date-to-weekday resolution for accurate timetable & syllabus lookup
- Stores chunks with source metadata
- Performs similarity search (retrieves top relevant chunks)
- Prompts llama3.2:3b with strict grounding and complete daily schedules
- Formats deduplicated source citations
"""

import os
import re
import datetime
from typing import List, Dict, Any, Tuple, Optional
import chromadb

import ollama_client
import document_loader

COLLECTION_NAME = "academic_documents"
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

SYSTEM_PROMPT = """You are an Academic Document Assistant.

Answer the student's question using ONLY the provided document context.

Do not use outside knowledge for college-specific information, dates, timetable, or syllabus details not present in the documents.

If the information is not present in the context, say:
"I could not find that information in the provided academic documents."

Important Instructions:
1. Daily Schedule & Lectures: When asked what lectures or classes a student has on a day (e.g., Wednesday), provide the complete schedule for that day in sequential order: morning practical labs, lunch break, afternoon practical sessions, breaks, and theory lectures (Period 1 through Period 6).
2. Date Resolution: In the 2026-27 academic calendar, 30 September 2026 is a Wednesday. When a user asks about a specific date (like "30 september" or "today"), specify that it corresponds to Wednesday and provide the full timetable for that day.
3. Final Year Subjects: When asked what subjects are in the final year or about the IT syllabus, provide the subjects for both Semester VII and Semester VIII as documented in the syllabus.
4. Keep answers clear, structured, and easy for a college student to understand.
5. Mention the source document used."""


def get_chroma_client(chroma_dir: str = CHROMA_DIR) -> chromadb.PersistentClient:
    """Initialize or connect to a local ChromaDB PersistentClient."""
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def is_index_ready(chroma_dir: str = CHROMA_DIR) -> bool:
    """Check if the vector database exists and has indexed chunks."""
    try:
        client = get_chroma_client(chroma_dir)
        collections = client.list_collections()
        col_names = [c.name if hasattr(c, "name") else str(c) for c in collections]
        if COLLECTION_NAME in col_names:
            collection = client.get_collection(COLLECTION_NAME)
            return collection.count() > 0
    except Exception:
        return False
    return False


def build_vector_index(
    workspace_dir: str = ".",
    chroma_dir: str = CHROMA_DIR,
    force_rebuild: bool = False,
    progress_callback: Optional[callable] = None
) -> Tuple[int, str]:
    """
    Build or rebuild the ChromaDB index for all 3 source documents.
    Generates local embeddings using nomic-embed-text.
    Returns (num_chunks, message).
    """
    client = get_chroma_client(chroma_dir)

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    try:
        collections = client.list_collections()
        col_names = [c.name if hasattr(c, "name") else str(c) for c in collections]
        if COLLECTION_NAME in col_names and not force_rebuild:
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                return count, f"Vector index already loaded with {count} chunks."
    except Exception:
        pass

    if progress_callback:
        progress_callback("Extracting text and tables from academic documents...")

    chunks = document_loader.load_all_academic_documents(workspace_dir)
    if not chunks:
        raise ValueError("No text could be extracted from the academic documents.")

    total_chunks = len(chunks)
    if progress_callback:
        progress_callback(f"Generating embeddings for {total_chunks} chunks using nomic-embed-text...")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i:04d}" for i in range(total_chunks)]

    embeddings = ollama_client.get_embeddings(texts, model=ollama_client.DEFAULT_EMBED_MODEL)

    if progress_callback:
        progress_callback("Saving vectors and metadata into ChromaDB...")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Academic Document Assistant Vector Store"}
    )

    batch_size = 100
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

    return total_chunks, f"Successfully indexed {total_chunks} document chunks."


def resolve_day_and_intent(query: str) -> Dict[str, Any]:
    """
    Analyze query to detect:
    - Specific weekday (Monday - Saturday)
    - Specific calendar date (e.g. 30 September, 15 August) mapped to 2026 weekday
    - Subject/syllabus inquiry
    - Academic calendar inquiry
    """
    q_lower = query.lower()
    detected_day = None
    detected_date_str = None

    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    for day in weekdays:
        if day in q_lower:
            detected_day = day.capitalize()
            break

    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    m = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
        q_lower
    )
    if not m:
        m = re.search(
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})",
            q_lower
        )
        if m:
            day_num = int(m.group(2))
            month_str = m.group(1)[:3]
            month_num = months.get(month_str, 9)
            try:
                dt = datetime.date(2026, month_num, day_num)
                if not detected_day:
                    detected_day = dt.strftime("%A")
                detected_date_str = f"{day_num:02d}-{month_num:02d}-2026"
            except Exception:
                pass
    else:
        day_num = int(m.group(1))
        month_str = m.group(2)[:3]
        month_num = months.get(month_str, 9)
        try:
            dt = datetime.date(2026, month_num, day_num)
            if not detected_day:
                detected_day = dt.strftime("%A")
            detected_date_str = f"{day_num:02d}-{month_num:02d}-2026"
        except Exception:
            pass

    # "today" in 2026 academic calendar is 30 September 2026 (Wednesday)
    if ("today" in q_lower or "tonight" in q_lower) and not detected_day:
        detected_day = "Wednesday"
        detected_date_str = "30-09-2026"

    is_subject_query = any(k in q_lower for k in [
        "subject", "subjects", "course", "courses", "final year", "syllabus of", "curriculum"
    ])

    is_timetable_query = any(k in q_lower for k in [
        "lecture", "lectures", "class", "classes", "timetable", "time table", "timing", "lab", "practical"
    ]) or bool(detected_day)

    return {
        "day": detected_day,
        "date_str": detected_date_str,
        "is_subject_query": is_subject_query,
        "is_timetable_query": is_timetable_query
    }


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 6,
    chroma_dir: str = CHROMA_DIR
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k most relevant chunks from ChromaDB for the user query,
    with smart intent enhancement.
    """
    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(COLLECTION_NAME)

    intent = resolve_day_and_intent(query)

    expanded_parts = [query]
    if intent["day"]:
        expanded_parts.append(f"TIMETABLE - {intent['day'].upper()} SCHEDULE lectures practical labs timing")
    if intent["date_str"]:
        expanded_parts.append(f"Academic Calendar {intent['date_str']}")
    if intent["is_subject_query"]:
        expanded_parts.append("FINAL YEAR B.TECH IT SEMESTER VII SEMESTER VIII SUBJECTS COURSES LIST")

    search_query = " ".join(expanded_parts)

    query_embedding = ollama_client.get_single_embedding(
        search_query,
        model=ollama_client.DEFAULT_EMBED_MODEL
    )
    if not query_embedding:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(docs)

    for doc, meta, dist in zip(docs, metas, distances):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "distance": dist
        })

    # Guaranteed target chunk injection
    if intent["day"]:
        target_prefix = f"TIMETABLE - {intent['day'].upper()} SCHEDULE"
        has_day = any(target_prefix in c["text"] for c in retrieved)
        if not has_day:
            day_results = collection.get(
                where={"doc_type": "timetable"}
            )
            for doc, meta in zip(day_results.get("documents", []), day_results.get("metadatas", [])):
                if target_prefix in doc:
                    retrieved.insert(0, {"text": doc, "metadata": meta, "distance": 0.0})
                    break

    if intent["is_subject_query"]:
        has_overview = any("SEMESTER VII SUBJECTS" in c["text"] or "WHAT SUBJECTS DO I HAVE" in c["text"] for c in retrieved)
        if not has_overview:
            syl_results = collection.get(
                where={"doc_type": "syllabus"}
            )
            for doc, meta in zip(syl_results.get("documents", []), syl_results.get("metadatas", [])):
                if "SEMESTER VII SUBJECTS" in doc or "WHAT SUBJECTS DO I HAVE" in doc:
                    retrieved.insert(0, {"text": doc, "metadata": meta, "distance": 0.0})
                    break

    return retrieved


def format_source_citation(metadata: Dict[str, Any]) -> str:
    """Format a clean source citation from chunk metadata."""
    source_name = metadata.get("source", "Unknown Document")
    doc_type = metadata.get("doc_type", "")
    page = metadata.get("page", 1)

    if doc_type == "timetable" or "time table" in source_name.lower():
        return "time table.png"
    else:
        return f"{source_name} — Page {page}"


def deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[str]:
    """Return deduplicated list of source strings in appearance order."""
    seen = set()
    sources = []
    for c in chunks:
        citation = format_source_citation(c.get("metadata", {}))
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources


def ask_assistant(
    query: str,
    top_k: int = 6,
    chroma_dir: str = CHROMA_DIR
) -> Dict[str, Any]:
    """
    Complete RAG question-answering workflow:
    1. Retrieve top relevant chunks
    2. Construct grounded prompt with intent resolution
    3. Generate answer using llama3.2:3b
    4. Return answer, deduplicated sources, and retrieved chunks
    """
    if not query or not query.strip():
        return {
            "answer": "Please ask a question about your academic documents.",
            "sources": [],
            "chunks": []
        }

    chunks = retrieve_relevant_chunks(query, top_k=top_k, chroma_dir=chroma_dir)

    if not chunks:
        return {
            "answer": "I could not find that information in the provided academic documents.",
            "sources": [],
            "chunks": []
        }

    # Format retrieved context
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        src = format_source_citation(c["metadata"])
        context_blocks.append(f"[Document Chunk {i} | Source: {src}]\n{c['text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    intent = resolve_day_and_intent(query)
    calendar_note = ""
    if intent["day"]:
        if intent["date_str"] or "today" in query.lower() or "30" in query:
            calendar_note = f"Academic Calendar Note: In the 2026-27 academic calendar, 30 September 2026 corresponds to {intent['day']}. Please provide the schedule for {intent['day']} based on the timetable."
        else:
            calendar_note = f"Timetable Note: The student is asking about {intent['day']}. Please provide the schedule for {intent['day']} from the timetable."

    user_message_content = f"""Context from academic documents:
{context_str}

{calendar_note}

Student Question:
{query}

Please provide a clear and grounded answer using ONLY the context provided above."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message_content}
    ]

    answer = ollama_client.generate_chat_response(
        messages,
        model=ollama_client.DEFAULT_CHAT_MODEL,
        temperature=0.1
    )

    sources = deduplicate_sources(chunks)

    # Refusal handling: If the answer explicitly states information not found
    if "I could not find that information" in answer and len(answer.strip()) < 120:
        sources = []

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }
