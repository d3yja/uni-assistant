from langgraph.types import interrupt


def human_review(answer: str) -> str:

    print("\n" + "=" * 60)
    print("Assistant Draft:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    while True:

        choice = input(
            "\nApprove this answer? (yes / no / edit): "
        ).strip().lower()

        if choice == "yes":
            return answer

        elif choice == "edit":

            print("\nEnter the revised answer:")

            edited = input("> ")

            return edited

        elif choice == "no":

            print("\nPlease provide the corrected answer:\n")

            edited = input("> ")

            return edited

        else:

            print("Invalid input. Please type yes, no, or edit.")