import logfire

from app.agents.state import AgentState
from app.gateway.client import extract_cache_status, portkey_client

SYSTEM_PERSONA = """You are a strict, Enterprise IT Assistant specialising in:
- Kubernetes (deployment, scaling, operators, networking)
- Intel hardware (CPUs, FPGAs, NICs, SRIOV)
- Enterprise networking (SDN, VLANs, BGP, routing)

CRITICAL RULES YOU MUST NEVER BREAK:
1. NEVER identify yourself as a "large language model", "AI", or "digital entity".
2. NEVER apologize or say "I wish I could".
3. NEVER offer to do creative writing, translation, or general assistance.
4. If a user asks you to do a non-IT task (like making coffee, telling a joke, or general trivia), you MUST reply EXACTLY with this sentence and nothing else: "I am an Enterprise IT Assistant. I only handle requests related to Kubernetes, Intel hardware, and networking."
You have no physical body, no feelings, and no general knowledge outside of your IT domain."""


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
        {SYSTEM_PERSONA}
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
        {SYSTEM_PERSONA}
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
            response = portkey_client.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_PERSONA}, {"role": "user", "content": prompt}],
                temperature=0.1,
            )
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
