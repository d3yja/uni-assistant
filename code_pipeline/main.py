# ============================================================
# main.py
# ============================================================

from graph import graph


# ============================================================
# WELCOME SCREEN
# ============================================================

def welcome():

    print("\n")
    print("=" * 70)
    print("           UNIVERSITY ASSISTANT")
    print("=" * 70)
    print()

    print("Welcome to the University Assistant!")

    print()

    print("I can help you with:")

    print("- Academic Calendar")
    print("- Attendance Policy")
    print("- Grading Policy")
    print("- Hostel Rules")
    print("- Laboratory Guidelines")
    print("- Contact Information")
    print("- Memory")
    print("- General University Questions")

    print()

    print("Type 'exit' anytime to quit.")

    print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

def run():

    # Display the welcome screen once
    welcome()

    while True:

        print()

        question = input("Ask a question: ").strip()

        # --------------------------------------------
        # Exit Program
        # --------------------------------------------

        if question.lower() in ["exit", "quit", "q"]:

            print("\nThank you for using the University Assistant!")
            print("Goodbye!\n")

            break

        # --------------------------------------------
        # Empty Question
        # --------------------------------------------

        if not question:

            print("Please enter a question.")

            continue

        # --------------------------------------------
        # Initial Graph State
        # --------------------------------------------

        initial_state = {

            "question": question,

            "intent": "",

            "selected_documents": [],

            "retrieved_documents": [],

            "answer": "",

            "needs_human_review": False,

        }

        # --------------------------------------------
        # Run the LangGraph
        # --------------------------------------------

        try:

            graph.invoke(initial_state)

        except Exception as e:

            print("\nAn error occurred while processing your request.")

            print(e)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    run()