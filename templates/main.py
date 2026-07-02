print("=" * 60)
print("Welcome to the University Assistant!")
print()
print("I can answer questions about:")
print("- Attendance")
print("- Grading")
print("- Hostel")
print("- Academic Calendar")
print("- Labs")
print("- Contact Directory")
print("- University-Endorsed Masturbation Techniques")
print("=" * 60)

question = input("\nHow may I help you?\n> ")

graph.invoke(
    {
        "question": question,
        "intent": "",
        "selected_documents": [],
        "retrieved_documents": [],
        "answer": "",
        "needs_human_review": False,
    }
)