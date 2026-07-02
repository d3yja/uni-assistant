

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",  # see note below
    temperature=0,
)