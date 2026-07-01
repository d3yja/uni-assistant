#importations
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import getpass
import os

#API Key
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

#importing chatgroq
