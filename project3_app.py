import streamlit as st
import os
import chromadb

from dotenv import load_dotenv

from google import genai

from langchain_google_genai import ChatGoogleGenerativeAI

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Company AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

h1, h2, h3 {
    font-weight: 700;
}

.small-text {
    font-size: 0.9rem;
    color: gray;
}

.source-box {
    padding: 12px;
    border-radius: 10px;
    background-color: rgba(200, 200, 200, 0.08);
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    padding-top: 30px;
    color: gray;
    font-size: 0.9rem;
}

</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

API_KEY = os.getenv("Course_AI_Lab")

if not API_KEY:

    st.error(
        "❌ Gemini API key not found.\n"
        "Please configure Course_AI_Lab "
        "inside your .env file."
    )

    st.stop()

# =========================================================
# INITIALIZE GEMINI CLIENT
# =========================================================

@st.cache_resource
def init_gemini_client():

    return genai.Client(api_key=API_KEY)

# =========================================================
# EMBEDDING FUNCTION
# =========================================================

def get_embedding(text):

    response = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values

# =========================================================
# INITIALIZE CHROMADB
# =========================================================

@st.cache_resource
def init_chromadb():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="company_documents"
    )

    return collection

# =========================================================
# INITIALIZE LLM
# =========================================================

@st.cache_resource
def init_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0
    )

# =========================================================
# INITIALIZE RESOURCES
# =========================================================

gemini_client = init_gemini_client()

collection = init_chromadb()

llm = init_llm()

# =========================================================
# RAG FUNCTION
# =========================================================

def get_rag_response(query, n_results=3):

    try:

        # Query embedding
        query_embedding = get_embedding(query)

        # Search vector DB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        documents = results["documents"][0]

        distances = results["distances"][0]

        metadatas = results["metadatas"][0]

        if not documents:

            return {
                "answer": (
                    "I couldn't find relevant information "
                    "in the knowledge base."
                ),
                "sources": [],
                "distances": []
            }

        # Build context
        context = "\n\n---\n\n".join(documents)

        # Prompt
        prompt = f"""
You are a professional HR AI Assistant.

Answer ONLY using the provided context.

If the answer is not present in the context,
say:
"I don't know based on the provided documents."

Provide concise and professional answers.

Context:
{context}

Question:
{query}

Answer:
"""

        # Generate response
        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": metadatas,
            "distances": distances
        }

    except Exception as e:

        return {
            "answer": f"⚠️ Error: {str(e)}",
            "sources": [],
            "distances": []
        }

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🏢 Company AI")

    st.markdown(
        """
### Intelligent HR Knowledge Assistant

This assistant uses:

- Gemini Embeddings
- ChromaDB Vector Search
- Retrieval-Augmented Generation (RAG)
- Gemini Flash LLM

to answer employee questions using the
company knowledge base.
"""
    )

    st.divider()

    st.success("✅ Knowledge Base Online")

    st.metric(
        label="Indexed Documents",
        value=collection.count()
    )

    st.metric(
        label="Model",
        value="Gemini 2.5 Flash"
    )

    st.metric(
        label="Embedding Model",
        value="Gemini Embedding"
    )

    st.divider()

    st.markdown("### Suggested Questions")

    suggested_questions = [
        "What is the vacation policy?",
        "Can employees work remotely?",
        "What are maternity leave benefits?",
        "How many paid leaves are allowed?",
        "What is the dress code policy?"
    ]

    for question in suggested_questions:

        if st.button(question):

            st.session_state.selected_question = question

    st.divider()

    if st.button("🗑️ Clear Chat History"):

        st.session_state.messages = []

        st.rerun()

# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 Company Knowledge Base AI Assistant")

st.caption(
    "Powered by Gemini + ChromaDB + RAG"
)

# =========================================================
# WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:

    welcome_message = """
👋 Welcome!

I am your AI-powered company assistant.

Ask me about:

- HR policies
- Leave policies
- Remote work
- Employee benefits
- Company procedures
- Workplace guidelines

I answer using the internal company knowledge base.
"""

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": welcome_message
        }
    )

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    avatar = "🤖" if message["role"] == "assistant" else "👤"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])

# =========================================================
# HANDLE SUGGESTED QUESTIONS
# =========================================================

if "selected_question" in st.session_state:

    prompt = st.session_state.selected_question

    del st.session_state.selected_question

else:

    prompt = st.chat_input(
        "Ask about company policies..."
    )

# =========================================================
# USER INPUT
# =========================================================

if prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(prompt)

    # Assistant response
    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🔍 Searching company knowledge base..."
        ):

            result = get_rag_response(prompt)

            answer = result["answer"]

            sources = result["sources"]

            distances = result["distances"]

            # Display answer
            st.markdown(answer)

            # Sources
            if sources:

                with st.expander("📚 Sources Used"):

                    for i, source in enumerate(sources):

                        similarity = round(
                            (1 - distances[i]) * 100,
                            2
                        )

                        st.markdown(
                            f"""
<div class="source-box">

<b>Source:</b>
{source.get("source", "Unknown")}

<br>

<b>Similarity:</b>
{similarity}%

</div>
""",
                            unsafe_allow_html=True
                        )

        # Store assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">

Built with ❤️ using
Gemini • ChromaDB • LangChain • Streamlit

</div>
""",
    unsafe_allow_html=True
)