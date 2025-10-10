import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY loaded:", bool(api_key))

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # you can also use "gemini-2.0-flash"
    temperature=0.3
)

# Simple test prompt
response = llm.invoke("Say hello from Gemini LangChain integration!")
print("Response:", response.content)
