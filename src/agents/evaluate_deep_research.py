# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)

from typing_extensions import Literal

from langgraph.types import Send

from src.agents.state import DeepSearchState, QueryGenerationState, ReflectionState

import os
from dotenv import load_dotenv
import openai

load_dotenv()
gemini_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = os.getenv('GEMINI_API_MODEL')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

openai.api_key = os.getenv('OPENAI_API_KEY')
openai_model = os.getenv('OPENAI_API_MODEL')


def patent_deep_evaluation(state: DeepSearchState) -> Literal["patent_research", "patent_deep_review"]:
    """
    LangGraph routing function that directs the next step in the patent research process
        - Manages the research loop by deciding whether to continue collecting information or finalize the summary, based on the predefined maximum number of iterations.
    :param state:
    :return:
    """
    if 'gpt' in state.llm:
        return patent_deep_evaluation_by_openai(state)
    elif 'gemini' in state.llm:
        return patent_deep_evaluation_by_gemini(state)


def patent_deep_evaluation_by_gemini(state: DeepSearchState) -> Literal["patent_research", "patent_deep_review"]:
    if not state.is_sufficient and state.research_loop_count <= state.max_research_loops:
        return "patent_research"
    else:
        return "patent_deep_review"


def patent_deep_evaluation_for_QA(state: DeepSearchState) -> Literal["patent_passage_search", "finalize_answer"]:
    if 'gpt' in state.llm:
        return patent_deep_evaluation_by_openai(state)
    elif 'gemini' in state.llm:
        return patent_deep_evaluation_by_gemini_(state)


def patent_deep_evaluation_by_gemini_(state: DeepSearchState) -> Literal["patent_passage_search", "finalize_answer"]:
    if not state.is_sufficient and state.research_loop_count <= state.max_research_loops:
        return "patent_passage_search"
    else:
        return "finalize_answer"


def patent_deep_evaluation_by_openai(state: DeepSearchState):
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else 3
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        print("")


def continue_to_patent_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the patent research agent.

    This is used to spawn n number of patent research nodes, one for each search query.
    """
    return [
        Send("patent_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]
