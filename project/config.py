# --------------------------------------------------------------------------
# Paths (used by the RAG chain)
# --------------------------------------------------------------------------
DOCS_DIR = "./project/docs"
PERSIST_DIR = "./db_md/chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# --------------------------------------------------------------------------
# Available documents: filename -> short description
# --------------------------------------------------------------------------
AVAILABLE_DOCS = {
    "academic_calendar.md": "Semester dates, exam schedules, holidays, and academic deadlines.",
    "grading_policy.md": "How grades are calculated, GPA rules, grade appeals, grading scales.",
    "attendance_policy.md": "Attendance requirements, absence rules, and consequences for missing classes.",
    "hostel_rules.md": "Hostel/dormitory rules, curfews, visitor policy, disciplinary procedures.",
    "contact_directory.md": "Contact info for faculty, staff, departments, and offices.",
}

# --------------------------------------------------------------------------
# Intents produced by classify_intent, matching the diamond's branches
# --------------------------------------------------------------------------
# INTENTS = {
#     "general_greeting": "Small talk / greetings that need only a simple canned reply.",
#     "memory_query": "The user is asking to recall or store something in conversation memory.",
#     "doc_query": "The question needs information looked up from the university documents.",
#     "human_needed": "The request needs a human, or a special interrupt-and-reply response.",
#     "unknown": "Intent is unclear and needs clarification from the user.",
# }

INTENTS = {
    "general_greeting": (
        "Greetings, introductions, thanks, or casual conversation that "
        "only requires a simple response."
    ),

    "memory_query": (
        "The user wants to store, update, or recall personal information "
        "from the conversation memory."
    ),

    "doc_query": (
        "The question requires retrieving factual information from the "
        "university knowledge base (RAG), such as attendance, grading "
        "policy, hostel rules, lab guidelines, academic calendar, "
        "scholarships, exam rules, or contact information."
    ),

    "human_needed": (
        "The request requires human review or approval, such as leave "
        "requests, permissions, personal academic decisions, or when the "
        "knowledge base cannot provide a reliable answer."
    ),

    "unknown": (
        "The user's intent is unclear and more clarification is needed."
    ),
}