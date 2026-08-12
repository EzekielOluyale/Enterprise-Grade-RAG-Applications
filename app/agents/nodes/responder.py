import logfire
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.agents.state import AgentState
from app.config import settings
from app.gateway.client import extract_cache_status, portkey_client


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are an Enterprise IT Assistant. You are strictly restricted to answering questions about:
        - Kubernetes (deployment, scaling, networking, operators)
        - Intel Hardware (CPUs, FPGAs, SRIOV, NICs)
        - Enterprise Networking (SDN, VLANs, BGP, routing)

        CRITICAL RULES:
        1. If the user asks about your capabilities ("what can you do", "who are you"), reply ONLY with:
           "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can help you with deployment, scaling, hardware specifications, and networking architectures. What can I help you with today?"
        2. If the user asks an off-topic or general question (such as finance, celebrities, math, coding outside these topics, etc.), you MUST refuse to answer. Say:
           "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!"
        3. if the user ask off topic question, reply ONLY with: I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can help you with deployment, scaling, hardware specifications, and networking architectures. What can I help you with today?

        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = _generate_response(prompt)
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}],
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _generate_response(prompt: str):
    """Call the LLM gateway with retry logic for transient failures."""
    return portkey_client.chat.completions.create(
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        messages=[{"role": "user", "content": prompt}],
    )
