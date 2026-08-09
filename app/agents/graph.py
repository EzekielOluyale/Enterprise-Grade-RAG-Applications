from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import AgentState

 # Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    # Initialize the State Graph
    workflow = StateGraph(AgentState)

    # Define the Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    workflow.set_entry_point("planner")

    # Conditional Edge: Planner -> Router -> (Retriever OR Responder)
    workflow.add_conditional_edges("planner", route_planner, {"retriever": "retriever", "responder": "responder"})

    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    # Compile the Graph with Memory
    return workflow.compile(checkpointer=checkpointer)