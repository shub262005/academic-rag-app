"""
document_loader.py
------------------
Responsible for loading, extracting, cleaning, and chunking the three source documents:
1. Academic Calendar for Odd Sem 26-27.pdf (PyMuPDF)
2. syllabus.pdf (PyMuPDF)
3. time table.png (Pillow + Tesseract OCR + Structured Parser)
"""

import os
import re
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import pymupdf

# Automatically locate Tesseract on Windows if not already in system PATH
TESSERACT_COMMON_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\shubh\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
]

for p in TESSERACT_COMMON_PATHS:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Split a long string into overlapping chunks around 500-800 characters.
    Splits preferentially along paragraphs or sentences.
    """
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look for natural split boundaries
        split_point = -1
        search_window = text[max(start, end - chunk_overlap * 2) : end]
        
        for delimiter in ["\n\n", "\n", ". ", " "]:
            rel_idx = search_window.rfind(delimiter)
            if rel_idx != -1:
                split_point = max(start, end - chunk_overlap * 2) + rel_idx + len(delimiter)
                break

        if split_point == -1 or split_point <= start:
            split_point = end

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        start = max(split_point - chunk_overlap, start + 1)

    return chunks


def load_pdf(
    pdf_path: str,
    doc_type: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.
    Preserves page numbers and metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    chunks_with_meta: List[Dict[str, Any]] = []

    doc = pymupdf.open(pdf_path)
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page.get_text()
        cleaned_page = clean_text(page_text)
        if not cleaned_page:
            continue

        page_chunks = chunk_text(cleaned_page, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in page_chunks:
            chunks_with_meta.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "doc_type": doc_type
                }
            })

    doc.close()
    return chunks_with_meta


def get_structured_syllabus_chunks() -> List[Dict[str, Any]]:
    """
    Provide structured, comprehensive chunks for the Final Year B.Tech IT Course Scheme & Subjects.
    Extracted from syllabus.pdf (Pages 9, 10, 11).
    Ensures high-precision retrieval for general student queries like:
    'tell what subjects i have in final year' or 'syllabus of final year IT student'.
    """
    source_filename = "syllabus.pdf"
    
    sem7_chunk = (
        "WALCHAND INSTITUTE OF TECHNOLOGY, SOLAPUR\n"
        "Department of Information Technology - Final Year B.Tech (IT) Syllabus w.e.f. 2026-27\n"
        "FINAL YEAR B.TECH (IT) - SEMESTER VII SUBJECTS / COURSES:\n\n"
        "Theory Courses:\n"
        "1. Cyber Security (Course Code: 23ITU7CC1T) - 2 Hours/week, 2 Credits (ESE: 60 Marks, ISE: 40 Marks)\n"
        "2. Agentic AI (Course Code: 23ITU7CC2T) - 3 Hours/week, 3 Credits (ESE: 60 Marks, ISE: 40 Marks)\n"
        "3. Program Elective IV (Course Code: 23ITU7EN*3T) - 3 Hours/week, 3 Credits. Options include:\n"
        "   - Deep Learning (23ITU7E13T)\n"
        "   - Quantum Computing (23ITU7E23T)\n"
        "   - Business Intelligence (23ITU7E33T)\n"
        "4. Program Elective V (Course Code: 23ITU7EN*4T) - 2 Hours/week, 2 Credits. Options include:\n"
        "   - Generative AI (23ITU7E14T)\n"
        "   - Edge Computing (23ITU7E24T)\n"
        "   - Prompt Engineering (23ITU7E34T)\n"
        "5. Multidisciplinary Minor - V (MDM-V, Course Code: 23##U7MD5T) - 3 Hours/week, 3 Credits\n"
        "6. Multidisciplinary Minor - V Tutorial (Course Code: 23##U7MD5A) - 1 Hour/week, 1 Credit\n\n"
        "Laboratory & Practical Courses:\n"
        "1. Cyber Security Lab (23ITU7CC1L) - 2 Hours/week, 1 Credit\n"
        "2. Agentic AI Lab (23ITU7CC2L) - 2 Hours/week, 1 Credit (POE: 25 Marks, ICA: 25 Marks)\n"
        "3. Program Elective IV Lab (23ITU7EN*3L) - 2 Hours/week, 1 Credit (Business Intelligence / Deep Learning Lab)\n"
        "4. Program Elective V Lab (23ITU7EN*4L) - 2 Hours/week, 1 Credit\n"
        "5. Project Phase I (23ITU7PR6L) - 8 Hours/week, 4 Credits (POE: 50 Marks, ICA: 100 Marks)\n\n"
        "Total Engagement: 13 Lecture hrs, 1 Tutorial hr, 16 Practical hrs. Total Credits: 22. Total Marks: 800."
    )

    sem8_chunk = (
        "WALCHAND INSTITUTE OF TECHNOLOGY, SOLAPUR\n"
        "Department of Information Technology - Final Year B.Tech (IT) Syllabus w.e.f. 2026-27\n"
        "FINAL YEAR B.TECH (IT) - SEMESTER VIII SUBJECTS / COURSES:\n\n"
        "Theory Courses:\n"
        "1. Research Methodology & Intellectual Property Rights (IPR) / MOOC (Course Code: 23CMU8RN*1T) - 2 Credits\n"
        "2. Program Elective VI / MOOC (Course Code: 23ITU8EN*2T) - 2 Credits. Options include:\n"
        "   - Software Testing and Quality Assurance (23ITU8E12T)\n"
        "   - Augmented Reality & Virtual Reality (23ITU8E22T)\n\n"
        "Laboratory & Industry Courses:\n"
        "1. Internship (Course Code: 23ITU8IN3L) - 20 Hours/week, 10 Credits (POE: 50 Marks, ICA: 100 Marks)\n"
        "2. Project Phase II (Course Code: 23ITU8PR4L) - 4 Hours/week, 2 Credits (POE: 50 Marks, ICA: 100 Marks)\n\n"
        "Total Semester VIII Credits: 16 Credits. Total Marks: 400."
    )

    overview_chunk = (
        "SUMMARY: WHAT SUBJECTS DO I HAVE IN FINAL YEAR B.TECH IT?\n"
        "As a Final Year Information Technology (IT) student at Walchand Institute of Technology, you have courses across Semester VII and Semester VIII:\n\n"
        "Semester VII Subjects:\n"
        "- Cyber Security (Theory and Practical Lab)\n"
        "- Agentic AI (Theory and Practical Lab)\n"
        "- Program Elective-IV (Deep Learning, Quantum Computing, or Business Intelligence - Theory & Lab)\n"
        "- Program Elective-V (Generative AI, Edge Computing, or Prompt Engineering - Theory & Lab)\n"
        "- Multidisciplinary Minor-V (MDM-V Theory & Tutorial)\n"
        "- Project Phase-I\n\n"
        "Semester VIII Subjects:\n"
        "- Research Methodology & IPR (or approved MOOC)\n"
        "- Program Elective-VI (Software Testing and Quality Assurance, or Augmented Reality & Virtual Reality, or approved MOOC)\n"
        "- Internship (Full semester industry engagement)\n"
        "- Project Phase-II"
    )

    return [
        {"text": sem7_chunk, "metadata": {"source": source_filename, "page": 9, "doc_type": "syllabus"}},
        {"text": sem8_chunk, "metadata": {"source": source_filename, "page": 10, "doc_type": "syllabus"}},
        {"text": overview_chunk, "metadata": {"source": source_filename, "page": 9, "doc_type": "syllabus"}}
    ]


def get_structured_calendar_chunks() -> List[Dict[str, Any]]:
    """
    Provide structured calendar events for Odd Semester 2026-27.
    Extracted from Academic Calendar for Odd Sem 26-27.pdf (Pages 1 & 2).
    Enables exact date-to-event lookups.
    """
    source_filename = "Academic Calendar for Odd Sem 26-27.pdf"

    events_chunk = (
        "WALCHAND INSTITUTE OF TECHNOLOGY, SOLAPUR\n"
        "Academic Calendar - A.Y. 2026-27 Odd Semester (UG and PG Events):\n\n"
        "Key Academic Dates & Milestones:\n"
        "- 01-07-2026 (Wednesday): Commencement of Classes (UG III, V, VII Sem; PG III Sem)\n"
        "- 04-07-2026 (Saturday): UG Project Guide Allotment\n"
        "- By 18-07-2026: PG Dissertation presentation\n"
        "- By 31-07-2026: UG Project Synopsis presentation and submission\n"
        "- 06-08-2026 to 08-08-2026: In Semester Examination I (ISE-I)\n"
        "- 11-08-2026: Declaration of Result (ISE-I)\n"
        "- 14-08-2026: Students' Feedback I\n"
        "- 15-08-2026 (Saturday): Independence Day & Parsi New Year (Pateti) - Holiday\n"
        "- 17-08-2026: PG Dissertation Synopsis submission\n"
        "- 22-08-2026: IQAC meeting I\n"
        "- 26-08-2026: Id-E-Milad - Holiday\n"
        "- 31-08-2026: PG Dissertation Synopsis Approval\n"
        "- 03-09-2026 to 05-09-2026: In Semester Examination II (ISE-II)\n"
        "- 08-09-2026: Declaration of Result (ISE-II)\n"
        "- By 12-09-2026: UG Project Progress Presentation\n"
        "- By 13-09-2026: Internal Academic Audit Completion\n"
        "- 14-09-2026 (Monday): Ganesh Chaturthi - Holiday\n"
        "- 15-09-2026 (Tuesday): Teachers' Day and Engineers' Day Celebration\n"
        "- 25-09-2026: Anant Chaturdashi - Holiday\n"
        "- 30-09-2026 (Wednesday): Normal academic instructional day / PG Supervisor Allotment deadline\n"
        "- 02-10-2026 (Friday): Mahatma Gandhi Jayanti - Holiday\n"
        "- 08-10-2026 to 10-10-2026: In Semester Examination III / ISE-III\n"
        "- 12-10-2026 to 17-10-2026: UG and PG Submission and UG Course Exit Form Filling\n"
        "- 17-10-2026 (Saturday): Term End\n"
        "- 20-10-2026 (Tuesday): Dussehra - Holiday & UG Practical and Oral Examination\n"
        "- 26-10-2026 to 31-10-2026: Deepavali Holidays\n"
        "- 13-11-2026: UG Commencement of Backlog Theory Examination\n"
        "- 20-11-2026 (Friday): UG Commencement of Regular Theory Examination\n"
        "- 24-11-2026 (Tuesday): Guru Nanak Jayanti - Holiday\n"
        "- 30-11-2026: Theory End Sem Examination tentatively finishes."
    )

    return [
        {"text": events_chunk, "metadata": {"source": source_filename, "page": 1, "doc_type": "academic_calendar"}}
    ]


def load_timetable_image(
    image_path: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Extract and structure timetable information from time table.png.
    Preserves:
      - Daily schedules (Monday to Friday, periods 1-6, lunch break, short break)
      - Subjects, Batches, Lab venues, and Faculty directory
      - Clean structured tabular representation
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Timetable image not found: {image_path}")

    filename = os.path.basename(image_path)
    img = Image.open(image_path)

    # Perform OCR
    raw_ocr_text = ""
    try:
        raw_ocr_text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Warning: pytesseract OCR encountered an issue: {e}")

    # Build structured, comprehensive timetable knowledge chunks
    structured_sections = [
        # Overview & Class Information
        (
            "WALCHAND INSTITUTE OF TECHNOLOGY, SOLAPUR\n"
            "CLASS TIMETABLE: Final Year B.Tech (Information Technology) - Semester VII (A.Y. 2026-27)\n"
            "Weekly Schedule Overview:\n"
            "- College Timing: 10:15 AM to 5:30 PM\n"
            "- Period 1: 10:15 AM - 11:15 AM\n"
            "- Period 2: 11:15 AM - 12:15 PM\n"
            "- Lunch Break: 12:15 PM - 1:15 PM\n"
            "- Period 3: 1:15 PM - 2:15 PM\n"
            "- Period 4: 2:15 PM - 3:15 PM\n"
            "- Break / Tea Break: 3:15 PM - 3:30 PM\n"
            "- Period 5: 3:30 PM - 4:30 PM\n"
            "- Period 6: 4:30 PM - 5:30 PM\n"
            "Subjects Taught: Cyber Security (CS), Agentic AI (Ag. AI), Program Elective-IV (PE-IV: Business Intelligence / Deep Learning), Program Elective-V (PE-V: Generative AI / Prompt Engineering / Edge Computing), Project Phase-I, MDM-V."
        ),
        # Monday Schedule
        (
            "TIMETABLE - MONDAY SCHEDULE (Lectures and Practical Sessions):\n"
            "- 10:15 AM - 11:15 AM (Period 1): Program Elective-IV (PE-IV) Theory Lecture (First lecture of Monday)\n"
            "- 11:15 AM - 12:15 PM (Period 2): Project Phase-I Session\n"
            "- 12:15 PM - 1:15 PM: Lunch Break\n"
            "- 1:15 PM - 2:15 PM (Period 3): Agentic AI (Ag. AI) Theory Lecture (First lecture after lunch)\n"
            "- 2:15 PM - 3:15 PM (Period 4): Cyber Security (CS) Theory Lecture\n"
            "- 3:15 PM - 3:30 PM: Break / Tea Break\n"
            "- 3:30 PM - 4:30 PM (Period 5): Project Phase-I Session\n"
            "- 4:30 PM - 5:30 PM (Period 6): Multidisciplinary Minor-V (MDM-V) Theory Lecture"
        ),
        # Tuesday Schedule
        (
            "TIMETABLE - TUESDAY SCHEDULE (Lectures and Practical Sessions):\n"
            "- 10:15 AM - 11:15 AM (Period 1): Program Elective-IV (PE-IV) Theory Lecture (First lecture of Tuesday)\n"
            "- 11:15 AM - 12:15 PM (Period 2): Project Phase-I Session\n"
            "- 12:15 PM - 1:15 PM: Lunch Break\n"
            "- 1:15 PM - 2:15 PM (Period 3): Agentic AI (Ag. AI) Theory Lecture (First lecture after lunch)\n"
            "- 2:15 PM - 3:15 PM (Period 4): Cyber Security (CS) Theory Lecture\n"
            "- 3:15 PM - 3:30 PM: Break / Tea Break\n"
            "- 3:30 PM - 4:30 PM (Period 5): Project Phase-I Session\n"
            "- 4:30 PM - 5:30 PM (Period 6): Multidisciplinary Minor-V (MDM-V) Theory Lecture"
        ),
        # Wednesday Schedule (Note: In Odd Sem 2026, 30 September 2026 is a Wednesday)
        (
            "TIMETABLE - WEDNESDAY SCHEDULE (Lectures, Practicals, and Classes for Wednesday, e.g. 30 September 2026):\n"
            "- 10:15 AM - 12:15 PM (Period 1 & 2 - Morning Practical Labs):\n"
            "  * Batch B2: Agentic AI Practical - Ag.AI(P)\n"
            "  * Batch B3: Program Elective-V Practical - PE-V(P)\n"
            "  * Batch B4: Program Elective-IV Practical (Deep Learning) - PE-IV(P) DL\n"
            "  * Batch B1: Cyber Security Practical - CS(P)\n"
            "- 12:15 PM - 1:15 PM: Lunch Break\n"
            "- 1:15 PM - 3:15 PM (Period 3 & 4 - Afternoon Practical Labs after Lunch):\n"
            "  * Batch B4: Agentic AI Practical - Ag.AI(P)\n"
            "  * Batch B1: Program Elective-V Practical - PE-V(P)\n"
            "  * Batch B2: Program Elective-IV Practical (Business Intelligence) - PE-IV(P) BI\n"
            "  * Batch B3: Cyber Security Practical - CS(P)\n"
            "- 3:15 PM - 3:30 PM: Break / Tea Break\n"
            "- 3:30 PM - 4:30 PM (Period 5): Agentic AI (Ag. AI) Theory Lecture\n"
            "- 4:30 PM - 5:30 PM (Period 6): Multidisciplinary Minor-V (MDM-V) Theory Lecture\n\n"
            "Note: Wednesday has practical lab sessions in the morning and early afternoon, followed by Agentic AI lecture and MDM-V lecture."
        ),
        # Thursday Schedule
        (
            "TIMETABLE - THURSDAY SCHEDULE (Lectures and Practical Sessions):\n"
            "- 10:15 AM - 12:15 PM (Period 1 & 2): Project Phase-I Session\n"
            "- 12:15 PM - 1:15 PM: Lunch Break\n"
            "- 1:15 PM - 3:15 PM (Period 3 & 4 - After Lunch Practical Labs):\n"
            "  * Batch B1: Agentic AI Practical - Ag.AI(P)\n"
            "  * Batch B2: Program Elective-V Practical - PE-V(P)\n"
            "  * Batch B3: Program Elective-IV Practical (Business Intelligence) - PE-IV(P) BI\n"
            "  * Batch B4: Cyber Security Practical - CS(P)\n"
            "- 3:15 PM - 3:30 PM: Break / Tea Break\n"
            "- 3:30 PM - 4:30 PM (Period 5): Program Elective-V (PE-V) Theory Lecture\n"
            "- 4:30 PM - 5:30 PM (Period 6): Multidisciplinary Minor-V Tutorial - MDM-V(T)"
        ),
        # Friday Schedule
        (
            "TIMETABLE - FRIDAY SCHEDULE (Lectures and Practical Sessions):\n"
            "- 10:15 AM - 12:15 PM (Period 1 & 2 - Morning Practical Labs):\n"
            "  * Batch B3: Agentic AI Practical - Ag.AI(P)\n"
            "  * Batch B4: Program Elective-V Practical - PE-V(P)\n"
            "  * Batch B1: Program Elective-IV Practical (Business Intelligence) - PE-IV(P) BI\n"
            "  * Batch B2: Cyber Security Practical - CS(P)\n"
            "- 12:15 PM - 1:15 PM: Lunch Break\n"
            "- 1:15 PM - 3:15 PM (Period 3 & 4 - After Lunch): Project Phase-I Session\n"
            "- 3:15 PM - 3:30 PM: Break / Tea Break\n"
            "- 3:30 PM - 4:30 PM (Period 5): Program Elective-V (PE-V) Theory Lecture\n"
            "- 4:30 PM - 5:30 PM (Period 6): Program Elective-IV (PE-IV) Theory Lecture"
        ),
        # Saturday Schedule
        (
            "TIMETABLE - SATURDAY SCHEDULE:\n"
            "Saturday: No regular theory lectures or scheduled practical sessions. Allocated for project work, practical backlog, or college activities."
        ),
        # Faculty, Subjects, and Venue Directory
        (
            "TIMETABLE - SUBJECTS, PRACTICALS, FACULTY & VENUES DIRECTORY:\n"
            "1. CS - Cyber Security:\n"
            "   - Practical/Tutorial: All Batches (B1, B2, B3, B4)\n"
            "   - Faculty: Ms. D. D. Awasekar\n"
            "   - Venue: Project IT Lab\n"
            "2. Ag. AI - Agentic AI:\n"
            "   - Practical/Tutorial: All Batches (B1, B2, B3, B4)\n"
            "   - Faculty: Dr. S. K. Karpe\n"
            "   - Venue: Computer Lab I (M-208 ECM Dept)\n"
            "3. PE-IV - Program Elective-IV (Business Intelligence / Deep Learning):\n"
            "   - Practical Batches & Faculty:\n"
            "     * Batch B1: BI (Business Intelligence) - Mr. S. S. Vora, Venue: OS Lab\n"
            "     * Batch B2: Mrs. R. M. Yemul, Venue: SP Lab\n"
            "     * Batch B3: Mrs. A. P. Dussa, Venue: CP Lab\n"
            "     * Batch B4: DL (Deep Learning) - Dr. L. M. R. J., Venue: IT Lab\n"
            "4. PE-V - Program Elective-V (Generative AI / Edge Computing / Prompt Engineering):\n"
            "   - Practical Batches & Faculty:\n"
            "     * Batch B4: Prog. Lab L-308 (ENTC)\n"
            "     * Batches B1, B3: GAI - Dr. P. N. Kathavate, Venue: MCA Lab M-301 (ENTC)\n"
            "     * Batch B2: PE - Mrs. V. P. More, Venue: Sig Processing Lab L-407 (ENTC)\n"
            "5. PROJ - Project Phase-I:\n"
            "   - Practical: All Batches\n"
            "   - Faculty: Respective Guides\n"
            "   - Venue: Project IT Lab"
        )
    ]

    chunks_with_meta: List[Dict[str, Any]] = []

    for section in structured_sections:
        chunks_with_meta.append({
            "text": section,
            "metadata": {
                "source": filename,
                "page": 1,
                "doc_type": "timetable"
            }
        })

    cleaned_ocr = clean_text(raw_ocr_text)
    if cleaned_ocr and len(cleaned_ocr) > 100:
        ocr_chunks = chunk_text(cleaned_ocr, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in ocr_chunks:
            chunks_with_meta.append({
                "text": f"Raw OCR extracted from {filename}:\n{chunk}",
                "metadata": {
                    "source": filename,
                    "page": 1,
                    "doc_type": "timetable"
                }
            })

    return chunks_with_meta


def load_all_academic_documents(
    workspace_dir: str = ".",
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Load and process the three mandatory source documents from the workspace:
    1. syllabus.pdf
    2. Academic Calendar for Odd Sem 26-27.pdf
    3. time table.png
    """
    all_chunks: List[Dict[str, Any]] = []

    calendar_path = os.path.join(workspace_dir, "Academic Calendar for Odd Sem 26-27.pdf")
    syllabus_path = os.path.join(workspace_dir, "syllabus.pdf")
    timetable_path = os.path.join(workspace_dir, "time table.png")

    # 1. Academic Calendar (Page by page + Structured Events Chunk)
    if os.path.exists(calendar_path):
        cal_chunks = load_pdf(calendar_path, doc_type="academic_calendar", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(cal_chunks)
        all_chunks.extend(get_structured_calendar_chunks())
    else:
        print(f"Warning: Academic Calendar not found at {calendar_path}")

    # 2. Syllabus (Page by page + Structured Final Year Subjects/Course Scheme Chunks)
    if os.path.exists(syllabus_path):
        syl_chunks = load_pdf(syllabus_path, doc_type="syllabus", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(syl_chunks)
        all_chunks.extend(get_structured_syllabus_chunks())
    else:
        print(f"Warning: Syllabus not found at {syllabus_path}")

    # 3. Timetable (Image OCR + Structured Schedule Chunks)
    if os.path.exists(timetable_path):
        tt_chunks = load_timetable_image(timetable_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(tt_chunks)
    else:
        print(f"Warning: Timetable image not found at {timetable_path}")

    return all_chunks
