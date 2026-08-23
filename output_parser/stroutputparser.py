from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Use a free, open model instead of gated Gemma
# Option 1: Mistral (Recommended - good performance)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95
)

# Option 2: Zephyr (Alternative - also good)
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.7,
#     top_p=0.95
# )

# Option 3: Phi-3 (Smaller, faster)
# llm = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.7
# )

model = ChatHuggingFace(llm=llm)

# Output parser for structured output
parser = StrOutputParser()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)

# Create chains with output parsing
chain1 = template1 | model | parser
chain2 = template2 | model | parser

# Execute the chains
result1 = chain1.invoke({'topic': 'black hole'})
print("=" * 50)
print("DETAILED REPORT:")
print("=" * 50)
print(result1)
print("\n")

result2 = chain2.invoke({'text': result1})
print("=" * 50)
print("SUMMARY:")
print("=" * 50)
print(result2)