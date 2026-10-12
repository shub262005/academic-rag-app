# Academic Document Assistant

Academic Document Assistant is a local Retrieval-Augmented Generation (RAG) chatbot that answers questions about a student's syllabus, timetable, and academic calendar. Everything runs 100% locally on your machine without relying on external cloud APIs or proprietary third-party services.

---

## Source Documents Used

The application reads strictly and exclusively from the three provided academic documents located in the project folder:

1. **`Academic Calendar for Odd Sem 26-27.pdf`**  
   Contains academic dates, semester start/end milestones, internal exams (ISE-I, ISE-II), submissions, holidays, and regular/backlog theory examination dates.
2. **`syllabus.pdf`**  
   Contains curriculum details, course outcomes, unit-wise breakdown, teaching and examination schemes, and credit hours for Final Year B.Tech Information Technology subjects (Cyber Security, Agentic AI, Business Intelligence, Deep Learning, Generative AI, etc.).
3. **`time table.png`**  
   Contains the weekly class timetable, period timings, day-by-day lecture/lab breakdown, batch allocations (B1, B2, B3, B4), faculty names, and lab venues.

No additional academic source documents are required or used.

---

## Technologies Used

- **Streamlit**: Fast, modern, responsive Python web interface for chat interaction, sidebar controls, and source presentation.
- **Ollama**: Local LLM and embedding runtime ensuring private, on-device AI inference.
- **`llama3.2:3b`**: Local quantized Large Language Model used to generate grounded, concise answers.
- **`nomic-embed-text`**: High-performance embedding model that converts document chunks and user queries into dense 768-dimensional semantic vectors.
- **ChromaDB**: Lightweight, embedded on-disk vector database storing document chunks and performing nearest-neighbor cosine similarity search.
- **PyMuPDF (`fitz`)**: Fast PDF parser used to extract page-by-page text from `syllabus.pdf` and `Academic Calendar for Odd Sem 26-27.pdf`.
- **Pillow & Pytesseract (OCR)**: Optical Character Recognition engine to extract and structure timetable information from `time table.png`.
- **RAG Architecture**: Retrieval-Augmented Generation pipeline combining vector similarity search with strict LLM system prompting to eliminate hallucinations.

---

## Project Structure

```text
D:\Project\Prompt Project\
├── app.py                                   # Streamlit frontend & session state management
├── rag.py                                   # RAG pipeline, ChromaDB store & grounded prompting
├── document_loader.py                       # PDF extraction, timetable OCR & chunking
├── ollama_client.py                         # Local Ollama HTTP API client (embeddings & chat)
├── requirements.txt                         # Python package dependencies
├── README.md                                # Setup guide and system documentation
├── chroma_db/                               # Local persistent ChromaDB vector store
├── Academic Calendar for Odd Sem 26-27.pdf  # Source academic calendar
├── syllabus.pdf                             # Source syllabus document
└── time table.png                           # Source timetable image
```

---

## System Requirements

1. **Python**: Version 3.10 or higher (Tested with Python 3.14).
2. **Ollama**: Installed and running locally on `http://localhost:11434`.
3. **Tesseract OCR (Windows)**:
   - Tesseract is already installed at: `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - If setting up on a new Windows system, download and install the installer from:
     `https://github.com/UB-Mannheim/tesseract/wiki`

---

## Setup & Installation

### Step 1: Pull Required Ollama Models

Open a terminal or PowerShell prompt and run:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Verify models are available:

```bash
ollama list
```

### Step 2: Install Python Dependencies

In the project folder, install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Streamlit will launch locally in your browser at:
`http://localhost:8501`

---

## How It Works (RAG Pipeline)

1. **Document Ingestion & OCR**:
   - `syllabus.pdf` and `Academic Calendar for Odd Sem 26-27.pdf` are extracted page-by-page using PyMuPDF with preserved page numbers.
   - `time table.png` is processed via OCR and cleaned into structured daily schedules and a faculty/venue directory.
   - Text is split into overlapping chunks of 500–800 characters with 100 characters overlap.
2. **Vector Indexing**:
   - `nomic-embed-text` generates 768-dimensional embeddings for all chunks.
   - Vectors and document metadata (`source`, `page`, `doc_type`) are stored locally in `chroma_db/`.
   - Index is built automatically on first launch and cached. It is only regenerated if missing or when clicking **Rebuild Document Index**.
3. **Grounded Query Answering**:
   - When a student asks a question, the query is embedded and the top 4 most relevant chunks are retrieved from ChromaDB.
   - The context and user query are passed to `llama3.2:3b` with a strict grounding prompt.
   - The assistant provides a concise answer along with exact, deduplicated source citations (e.g., `syllabus.pdf — Page 26`, `time table.png`).
   - If the answer is not present in the documents, the model responds:  
     `"I could not find that information in the provided academic documents."`
# academic-rag-app
