from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.graph import route_research
from src.agents.passage_chat_graph import route_chat_research
from src.agents.state import DeepSearchState, DeepSearchStateInput, DeepSearchStateOutput

# Define the system prompt instructing the agent on how to answer the user's question.
SYSTEM_PROMPT = """You will act as a patent expert for analysing patents and perform a deep research"""

# Define the user's initial question about exchange rates.
USER_PROMPT = """How does the application of cold plasma technology in agriculture affect seed germination rates and plant growth?"""


graph = route_chat_research(state=DeepSearchState)
# Initialize a LangGraph thread with a unique ID for state management.
thread_config = {"configurable": {"thread_id": "1"}}

# Execute the LangGraph workflow, streaming the results of each node.
for state in graph.stream(
        {
            "research_topic": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=USER_PROMPT),
            ],
        },
        thread_config,
):
    # Print the name of the current node and its output for each step.
    for node_name, node_output in state.items():
        print(f"Agent Node: {node_name}\n")
        print("Agent Result:")
        print(str(node_output))  # Truncate output for display
    print("\n====================\n")
