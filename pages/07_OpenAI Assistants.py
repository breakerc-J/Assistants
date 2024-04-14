import streamlit as st
import os
from typing import Type
from pydantic import BaseModel
from pydantic import Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnablePassthrough



class WikipediaSearchAgent(BaseTool):
    name = "WikipediaSearchAgent"
    description = """
    Use this tool to find the website for the given query.
    """
    class WikipediaSearchAgentArgsSchema(BaseModel):
        query: str = Field(
        )
    args_schema: Type[WikipediaSearchAgentArgsSchema] = WikipediaSearchAgentArgsSchema
    def _run(self, query):
        return WikipediaAPIWrapper().run(query)


class DuckDuckGoSearchAgent(BaseTool):
    name = "DuckDuckGoSearchAgent"
    description = """
    Use this tool to find the website for the given query.
    """
    class DuckDuckGoSearchAgentArgsSchema(BaseModel):
        query: str = Field(
        )
    args_schema: Type[DuckDuckGoSearchAgentArgsSchema] = DuckDuckGoSearchAgentArgsSchema
    def _run(self, query):
        return DuckDuckGoSearchAPIWrapper().run(query)


class LoadWebsiteTool(BaseTool):
    name = "LoadWebsiteTool"
    description = """
    Use this tool to load the website for the given url.
    """
    class LoadWebsiteToolArgsSchema(BaseModel):
        url: str = Field(
        )
    args_schema: Type[LoadWebsiteToolArgsSchema] = LoadWebsiteToolArgsSchema
    def run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        return docs


class SaveToFileTool(BaseTool):
    name = "SaveToFileTool"
    description = """
    Use this tool to save the .txt file.
    """
    class SaveToFileToolArgsSchema(BaseModel):
        text: str = Field(
        )
        file_path: str = Field(
        )
    args_schema: Type[SaveToFileToolArgsSchema] = SaveToFileToolArgsSchema
    
    def run(self, text, file_path):
        os.makedirs("./outputs", exist_ok=True)
        with open(f"{file_path}", "w", encoding="utf-8") as f:
            f.write(text)

        return f"Text saved to {file_path}"



st.title("OpenAI Assistants")
st.markdown(
    """
    This is research AI agent.
            
    The agent should try to search in Wikipedia or DuckDuckGo about "Research about the XZ backdoor" and extract it's content, then it should finish by saving the research to a .txt file.

    """
)
api_key = st.session_state.get("api_key", "")

llm = ChatOpenAI(
    temperature=0.1,
    api_key=api_key,
)

with st.sidebar:
    api_key = st.text_input("OpenAI_API_key", type="password")
    st.session_state["api_key"] = api_key
    if api_key:
        st.caption("API key is set.")
    else:
        st.caption("Please enter your API key ⬆️.")

if api_key == "":
    st.error("Please enter your OpenAI API key")
    st.stop()
else:
    llm = ChatOpenAI(
        temperature=0.1,
        api_key=api_key,
        streaming=True,
    )

st.subheader("Run your OpenAI Assistants :)")
guide1, guide2 = st.columns([4, 1])
with guide1:
    query = st.text_input(
        "Subject",
        key="query_input",
        value="Research about the XZ backdoor",
        label_visibility="collapsed",
    )
with guide2:
    run_agent = st.button(
        "Run",
        key="run_button",
        type="primary",
        use_container_width=True,
    )


def agent_invoke(input):
    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        tools=[
            WikipediaSearchAgent(),
            DuckDuckGoSearchAgent(),
            LoadWebsiteTool(),
            SaveToFileTool(),
        ],
    )

    prompt = PromptTemplate.from_template(
        """
        You are a great search engine and business assistant.
        You just need to handle your work in the following order.
        1. Search for the information about a query in Wikipedia.
        2. Search for the information about a query in DuckDuckGo.
        3. If there is a list of website URLs in the search result list, the contents of each website are extracted as text.
        4. Save it as a .txt file.
        
        query: {query}    
        """,
    )
    chain = {"query": RunnablePassthrough()} | prompt | agent
    result = chain.invoke(input)
    return result["result"]

with st.sidebar:
    st.write(
        "https://github.com/breakerc-J/Assistants/commit/e60427b53c60b89522ebe49fd7337eb57d7dcd48"
    )
