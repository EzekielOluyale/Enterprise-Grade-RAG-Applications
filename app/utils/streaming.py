import json
import logfire

STATUS_MESSAGES = {
    "planner": "Planning execution...",
    "retriever": "Searching knowledge base...",
    "responder": "Generating answer...",
}

def format_sse(event_type: str, content: any = None, event_id: str = None) -> str:
    """
    Safely formats data for Server-Sent Events (SSE).
    """
    payload = json.dumps({
        "type": event_type,
        "content": content,
    })

    if event_id is None:
        return f"data: {payload}\n\n"

    return (
        f"id: {event_id}\n"
        f"data: {payload}\n\n"
    )

async def stream_agent(agent_instance, initial_state: dict, config: dict, thread_id: str):
    """
    Standalone generator function that executes the LangGraph workflow 
    and yields SSE formatted chunks.
    """
    logfire.info(f"🌊 Stream started | thread={thread_id}")
    
    try:
        # Initial connection status
        yield format_sse("status", "Agent started.")

        # Note on Heartbeats: SSE natively supports keeping connections alive 
        # by sending a comment string like `: keep-alive\n\n`. The proxy headers 
        # below usually handle this, but you can inject a ping here if needed.

        async for event in agent_instance.astream_events(initial_state, config=config, version="v2"):
            kind = event["event"]
            name = event.get("name", "")

            # 1. Dynamic Status Tracking via Dictionary
            if kind == "on_chain_start":
                message = STATUS_MESSAGES.get(name)
                if message:
                    yield format_sse("status", message)

            # 2. Token Extraction
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content"):
                    content = chunk.content
                    
                    if isinstance(content, str):
                        if content:
                            yield format_sse("token", content)
                            
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                yield format_sse("token", item["text"])
                            else:
                                text = getattr(item, "text", None)
                                if text:
                                    yield format_sse("token", text)

        # 3. Extract State and emit Metadata
        final_state = await agent_instance.aget_state(config)
        documents = final_state.values.get("documents", [])
        
        yield format_sse("metadata", {
            "sources": documents,
            "thread_id": thread_id
        })
        
        # Signal successful completion
        yield format_sse("end")
        logfire.info(f"✅ Stream completed successfully | thread={thread_id}")

    except Exception as e:
        # logfire.exception captures the full stack trace, not just the error string
        logfire.exception("Stream execution failed.")
        yield format_sse("error", "An internal error occurred during processing.")
        yield format_sse("end")
        
    finally:
        logfire.info(f"🔌 Stream connection closed | thread={thread_id}")