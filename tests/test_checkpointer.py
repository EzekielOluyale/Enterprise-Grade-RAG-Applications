"""Tests for LangGraph checkpointer configuration."""

from langgraph.checkpoint.memory import MemorySaver

from app.agents.graph import build_graph, route_planner

def test_route_planner_sends_conversational_to_responder():
    """Conversational queries should bypass the retriever and go straight to the responder."""
    state = {"current_query": "CONVERSATIONAL"}
    
    route = route_planner(state)
    
    assert route == "responder"


def test_route_planner_sends_other_queries_to_retriever():
    """Any other query should go through the retriever first."""
    state = {"current_query": "What is our company's refund policy?"}
    
    route = route_planner(state)
    
    assert route == "retriever"

def test_build_graph_uses_provided_checkpointer():
    """build_graph should compile using the supplied checkpointer."""
    saver = MemorySaver()
    agent = build_graph(checkpointer=saver)
    
    assert agent is not None
    assert agent.checkpointer is saver


def test_build_graph_works_without_checkpointer():
    """build_graph should successfully compile even if no checkpointer is provided."""
    agent = build_graph(checkpointer=None)
    
    assert agent is not None
    assert agent.checkpointer is None