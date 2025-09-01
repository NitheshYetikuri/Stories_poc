# === Imports ===
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
import chromadb
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

import os
load_dotenv()

# Load secrets from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY1 = os.environ.get("GOOGLE_API_KEY1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY1 = os.environ.get("GROQ_API_KEY1")
from crewai_tools import FileReadTool
file_read_tool = FileReadTool()
chroma_client = chromadb.PersistentClient(path="vectordb/chromadb/")
chroma_collection = chroma_client.get_or_create_collection(name="relations")
api = GOOGLE_API_KEY


@tool("retriever_tool")
def my_retriever_tool(question: str) -> str:
    "acts as json retriever using ChromaDB"
    db=Chroma(client=chroma_client, collection_name="relations", embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        api_key=api
    ))

    system_prompt = (
    "You are an expert Java backend developer for a Spring Boot project. "
    "Based on the user's query, identify which classes are directly or indirectly affected by this change, in standard Spring Boot microservice architecture."
    "Include the complete class path, which is used to find the path of the class in the local system."
    "Ensure there are no duplicates and do not include source code."
    "Only include paths and class names from the context provided by the retriever."
    "#important:always include complete source file path."
    "Also include required changes for classes"
    "Only include the class names and their complete file paths, for example: 'C:\\Users\\ with complete path from c folder to source and class"
   "expected Output classname:"
   "class path:"
   "changes:"
    "{context}"

        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":5})

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.3, max_tokens=2000)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": question})

    return response["answer"]

tool_retriever = my_retriever_tool
llm= LLM(model='gemini/gemini-2.5-flash', api_key=GOOGLE_API_KEY1)
llm_new = LLM(model='gemini/gemini-2.5-flash', api_key=GOOGLE_API_KEY)
llm_groq= LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY1)
# === Agent: Analyzer ===
project_analyzer_agent = Agent(
    role="Information Retrieval Expert",
    goal="Retrieve and list all impacted classes and their paths based on the user's query.",
    backstory=(
        "You are an information retrieval expert in a Java Spring Boot project. "
        "Given a user's query, you will retrieve relevant information about the impacted classes, "
        "analyze it, and refine the query if necessary to ensure complete results. "
        "Only list the class names and their paths without source code."
    ),
    tools=[tool_retriever],
    llm=llm_new,

    allow_delegation=False,
    verbose=True
)

query_response_task = Task(
    agent=project_analyzer_agent,
    description=(
        "Process the user's query: '{{user_query}}' by invoking the 'retriever_tool'. "
        "After retrieving initial results, analyze for missing information. "
        "and return the final impacted classes and their source file paths\n\n"
        "Format your final answer like this:\n\n"
        "[\n" "PATHS \n"
        "\"path/to/Class1.java\",\n"
        "\"path/to/Class2.java\",\n" "Changes \n"
        "change 1\n"
        "change 2\n"
        "... \n]"
    ),
    expected_output="The final output must be a valid Python list assignment containing file paths.",
    input={"user_query": "{{user_query}}"},
    output_file="project_output/paths.txt"
)
file_reader_agent = Agent(
    role="Java File Reader",
    goal="Read Java files from local and concatenate their content into a single output.",
    backstory="You fetch multiple Java source codes from local system and combine them.",
    tools=[file_read_tool],
    llm=llm,
    allow_delegation=False,
    verbose=True
)

file_reading_task = Task(
    agent=file_reader_agent,
    description=(
        "Given the  list of file paths from the previous task, "
        "read each file's content from GitHub and combine them."
    ),
    expected_output="Concatenated Java source code from all file paths.",
    context=[query_response_task],
    output_file="project_output/code.txt"
)
tech_lead_agent = Agent(
    role="Tech Lead",
    goal="Generate comprehensive user stories for developers and testers based on code changes and user requirements.",
    backstory="You are a highly experienced Tech Lead in a software company, skilled at translating feature requests into actionable user stories.",
    llm=llm_groq,
    allow_delegation=False,
    verbose=True
)

generate_stories_task = Task(
    description=(
        "Analyze the content of the provided combined Java code and previous task context and user query:{{user_query}}. "
        "Based on this analysis, generate two sets of user stories: one for developers and one for testers."
    ),
    agent=tech_lead_agent,
    expected_output="A string containing Developer and Tester user stories, clearly labeled.",
    context=[file_reading_task, query_response_task],
    output_file="project_output/stories.txt"
)

project_analysis_crew = Crew(
    agents=[project_analyzer_agent, file_reader_agent, tech_lead_agent],
    tasks=[query_response_task, file_reading_task, generate_stories_task],
    process=Process.sequential
)
def run_story_generation(query:str) -> str:

    result = project_analysis_crew.kickoff(inputs={
        "user_query": query
    })
    return result

# print(run_story_generation("add a feature to retrieve all order details within a price range"))