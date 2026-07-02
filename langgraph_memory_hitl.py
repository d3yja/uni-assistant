"""
============================================================================
 LangGraph Teaching Example: MEMORY + HUMAN-IN-THE-LOOP
============================================================================

USE CASE (kept deliberately simple):
    A "reply assistant". You type a message, the LLM drafts a reply, and the
    graph PAUSES so a human can approve or edit that draft before it becomes
    final. Because the whole conversation is saved, the assistant REMEMBERS
    earlier turns.

This single example demonstrates the two concepts students need:

  1. MEMORY  -> a "checkpointer" + a "thread_id" save graph state between
               runs, so the agent recalls the earlier conversation.

  2. HUMAN-IN-THE-LOOP (HITL) -> interrupt() pauses the graph and hands
               control to a human; Command(resume=...) feeds the human's
               decision back in and execution continues from where it stopped.

----------------------------------------------------------------------------
 SETUP (all free, no paid services):
   pip install langgraph langchain-groq
   Create a FREE key at https://console.groq.com  (no credit card needed)
   export GROQ_API_KEY="your_key_here"      # Windows: set GROQ_API_KEY=...
----------------------------------------------------------------------------
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver   # in-memory = free
from langgraph.types import interrupt, Command


# ---------------------------------------------------------------------------
# 1. THE LLM (free, via Groq Cloud)
# ---------------------------------------------------------------------------
# ChatGroq automatically reads the GROQ_API_KEY environment variable.
# 'llama-3.1-8b-instant' is a fast model on Groq's free tier.
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)


# ---------------------------------------------------------------------------
# 2. THE GRAPH STATE
# ---------------------------------------------------------------------------
# 'messages' is the conversation. The add_messages reducer APPENDS new
# messages instead of overwriting them -> this list is what gives us memory.
# 'draft' temporarily holds the AI's proposed reply while a human reviews it.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str


# ---------------------------------------------------------------------------
# 3. NODES (the steps in our workflow)
# ---------------------------------------------------------------------------
def draft_reply(state: State):
    """Ask the LLM to draft a reply using the FULL conversation history.

    Because the checkpointer restored every earlier message into
    state['messages'], the model sees the whole context = MEMORY in action.
    We store the draft separately; it is NOT added to the conversation yet,
    because a human still has to approve it.
    """
    ai_message = llm.invoke(state["messages"])
    return {"draft": ai_message.content}


def human_review(state: State):
    """Pause the graph and wait for a human decision (HUMAN-IN-THE-LOOP).

    interrupt() stops execution right here and surfaces this payload to the
    caller. The graph's state is saved by the checkpointer, so it can wait
    indefinitely. When we later resume with Command(resume=<value>), that
    <value> becomes the return value of interrupt() and the function runs on.
    """
    human_decision = interrupt({
        "proposed_reply": state["draft"],
        "instructions": "Type 'ok' to approve, or type your own edited reply.",
    })

    # Decide what the FINAL reply should be, based on the human's input.
    if human_decision.strip().lower() == "ok":
        final_reply = state["draft"]          # human approved as-is
    else:
        final_reply = human_decision          # human edited it

    # Only now is the approved reply committed to the conversation memory.
    return {"messages": [AIMessage(content=final_reply)], "draft": ""}


# ---------------------------------------------------------------------------
# 4. BUILD & COMPILE THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(State)
builder.add_node("draft_reply", draft_reply)
builder.add_node("human_review", human_review)

builder.add_edge(START, "draft_reply")        # start -> draft a reply
builder.add_edge("draft_reply", "human_review")  # -> pause for the human
builder.add_edge("human_review", END)         # -> finished this turn

# The checkpointer is what makes BOTH features work:
#   * memory: it persists state between turns
#   * HITL:   it saves state during an interrupt so we can resume later
graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# 5. RUN IT (a short interactive chat loop)
# ---------------------------------------------------------------------------
def main():
    # The thread_id is the "memory key". Reusing the SAME id across turns
    # tells LangGraph to load this conversation's saved history each time.
    config = {"configurable": {"thread_id": "chat-1"}}

    print("Reply assistant (type 'quit' to exit)\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break

        # --- Run the graph until it hits the interrupt() ---
        # Passing only the new message is enough; add_messages + the
        # checkpointer merge it with the remembered history automatically.
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config,
        )

        # When a graph pauses, the result contains an "__interrupt__" key
        # holding the payload we passed to interrupt().
        payload = result["__interrupt__"][0].value
        print("\n--- AI DRAFT (awaiting your approval) ---")
        print(payload["proposed_reply"])
        print(f"({payload['instructions']})")

        # --- Collect the human decision and RESUME the graph ---
        decision = input("Your review: ")
        final = graph.invoke(Command(resume=decision), config)

        # The last message is the approved reply now stored in memory.
        print(f"\nAssistant: {final['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
