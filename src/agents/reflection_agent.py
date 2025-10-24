# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from src.agents.state import DeepSearchState, QueryGenerationState, ReflectionState

from typing import List
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
import openai

load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


def patent_deep_reflection(state: DeepSearchState) -> ReflectionState:
    if 'gpt' in state.llm:
        return patent_deep_reflection_by_openai(state)
    elif 'gemini' in state.llm:
        return patent_deep_reflection_by_gemini(state)


def patent_deep_reflection_by_gemini(state: DeepSearchState)  -> DeepSearchState:
    """
    reflection agent
    Args:
        state:

    Returns:

    """
    #topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    patent_reflection_prompt = f"""
                You are an expert research assistant analyzing patent summaries about the question or research topic '''{state.research_topic}'''
                
                Instructions:
                    - Detect knowledge gaps or areas requiring further exploration, and generate only one follow-up query.
                    - If the provided patent summaries fully address the user’s question, do not create additional queries.
                    - When gaps exist, formulate a follow-up query that deepen or broaden understanding.

                 Requirements:
                    - Ensure the follow-up query is self-contained and includes necessary concepts for patent search.
                    - provide only one query and should take the form of a natural-language sentence.
                    - Do not include any irrelevant information outside the topic/question '''{state.research_topic}'''.
                    - Keep it concise.
                    - Do not hallucinate.
                    
            Output Format:
                - Format your response as a JSON object with these exact keys:
                - "is_sufficient": true or false 
                - "knowledge_gap": Describe what information is missing or needs clarification 
                - "follow_up_query": Write a specific question to address this gap.
                
            Example:
            '''json
            {{
                "is_sufficient": true, or false
                "knowledge_gap": "The patent summary lacks information about examples of core inventions",  "" if is_sufficient is true
                "follow_up_query": "What are core inventions for [specific technology]?", "" if is_sufficient is true
            }}
            ''' 
            Reflect carefully on the patent Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output in JSON format:   
            PATENT SUMMARIES: '''{state.patent_research_results}''' \n
            """
    llm_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        temperature=0.5,
        max_retries=2,
        api_key=gemini_api_key)

    result = llm_model.with_structured_output(Reflection).invoke(patent_reflection_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_query": result.follow_up_queries,
        "research_loop_count": state.research_loop_count + 1,
        "patent_search_query": ' '.join(result.follow_up_queries),
    }


def patent_deep_reflection_by_openai(state: DeepSearchState)  -> ReflectionState:
    topic = [msg.content for msg in state.research_topic if isinstance(msg, HumanMessage)][0]
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    patent_reflection_prompt = f"""
                You are an expert research assistant analyzing patent passages about '''{topic}'''
                
                Instructions:
                    - Detect knowledge gaps or inventions requiring further exploration, and generate only one follow-up query.
                    - If the provided patent summaries fully address the user’s question, do not create any query.
                    - When gaps exist, formulate a follow-up query that deepen or broaden understanding.

                Requirements:
                - Ensure the follow-up query is self-contained and includes necessary concepts for patent search.
                - provide only one query and should take the form of a natural-language sentence.
                - Do not include any irrelevant information to the topic/question '''{topic}'''.
                - Keep it concise.
                - Do not hallucinate.
                   
            Output Format:
                - Format your response as a JSON object with these exact keys:
                - "is_sufficient": true or false 
                - "knowledge_gap": Describe what information is missing or needs clarification 
                - "follow_up_query": Write a specific query to address this gap.
                
            Example:
            '''json
            {{
                "is_sufficient": true, or false
                "knowledge_gap": "The provided summaries lack to details of the specific mechanism, device, method, or system",  "" if is_sufficient is true
                "follow_up_query": "What are specific mechanisms, devices, methods, or systems for [specific technology]?", "" if is_sufficient is true
            }}
            ''' 
            Reflect carefully on the patent Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output in JSON format:   
            PATENT SUMMARIES: '''{state.patent_running_summary}''' \n
            """
    llm_model = ChatOpenAI(
        model_name=openai_model,
        temperature=0.5,
    )

    result = llm_model.with_structured_output(Reflection).invoke(patent_reflection_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_query": result.follow_up_queries,
        "research_loop_count": state.research_loop_count + 1,
        "patent_search_query": ' '.join(result.follow_up_queries),
    }


def continue_to_patent_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the patent research agent.

    This is used to spawn n number of patent research nodes, one for each search query.
    """
    return [
        Send("patent_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]
