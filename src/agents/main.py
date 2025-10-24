# Copyright 2025 FIZ-Karlsruhe (Mustafa Sofean)


from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.graph import route_research
from src.agents.state import DeepSearchState

from src.agents.passage_chat_graph import route_chat_research

def run_deep_research(system_prompt, user_prompt, llm):
    results = {}
    graph = route_research(state=DeepSearchState)
    # Initialize a LangGraph thread with a unique ID for state management.
    thread_config = {"configurable": {"thread_id": "1"}}
    # Execute the LangGraph workflow, streaming the results of each node.
    for state in graph.stream(
            {
                "research_topic": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                "llm": llm
            },
            thread_config,
    ):
        for node_name, node_output in state.items():
            results[node_name] = node_output

    return results


def run_chat_deep_research(system_prompt, user_prompt, llm):
    results = {}
    graph = route_chat_research(state=DeepSearchState)
    # Initialize a LangGraph thread with a unique ID for state management.
    thread_config = {"configurable": {"thread_id": "1"}}
    # Execute the LangGraph workflow, streaming the results of each node.
    for state in graph.stream(
            {
                "research_topic": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
            },
            thread_config,
    ):
        for node_name, node_output in state.items():
            results[node_name] = node_output

    return results

