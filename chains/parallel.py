from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

model1=ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7,
    max_output_tokens=512,
    api_key=os.getenv("GOOGLE_API_KEY")
)

model2=ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7,
    max_output_tokens=512,
    api_key=os.getenv("GOOGLE_API_KEY")
)
model3=ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0.7,
    max_output_tokens=512,
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',   
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',  
    input_variables=['text']
)


parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'summary': prompt2 | model2 | parser
})

result = parallel_chain.invoke({'topic': 'Unemployment in India'})  

print(result)


