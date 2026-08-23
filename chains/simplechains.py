from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",  # or "gemini-1.5-flash" for faster/cheaper
    temperature=0.7,
    max_output_tokens=512,
    api_key=os.getenv("GOOGLE_API_KEY")  # Get API key from .env
)

# Create prompt
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Create parser
parser = StrOutputParser()

# Create chain
chain = prompt | model | parser

# Run the chain
result = chain.invoke({'topic': 'cricket'})

print("=" * 60)
print("FACTS ABOUT CRICKET:")
print("=" * 60)
print(result)
print("\n" + "=" * 60)

# Print chain structure
chain.get_graph().print_ascii()