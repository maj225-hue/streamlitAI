
import streamlit as st          # Creates web interface components
from pathlib import Path
import tempfile
import io
from docling.document_converter import DocumentConverter

# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- SESSION STATE INITIALIZATION ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'converted_docs' not in st.session_state:
    st.session_state.converted_docs = []
if 'collection' not in st.session_state:
    st.session_state.collection = None

# --- File Upload and Conversion ---
def convert_and_store(files):
    """
    Convert uploaded files to plain text and return as a list.
    Args:
        files (list): List of uploaded files from Streamlit file_uploader.
    Returns:
        list: List of document texts extracted from files.
    """
    converter = DocumentConverter()
    documents = []
    for uploaded_file in files:
        try:
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                st.error(f"{uploaded_file.name} is too large! Skipping.")
                continue
            # Use delete=False for Windows compatibility
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            doc = converter.convert(tmp_path).document
            if hasattr(doc, 'text') and doc.text:
                documents.append(doc.text)
            elif hasattr(doc, 'export_to_markdown'):
                documents.append(doc.export_to_markdown())
            else:
                documents.append(str(doc))
            Path(tmp_path).unlink()  # Clean up temp file
        except Exception as e:
            st.warning(f"Could not convert {uploaded_file.name}: {e}")
    return documents
# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

def setup_collection_from_upload(docs):
    """
    Stores uploaded/converted documents in a fresh ChromaDB collection (removes any previous collection).
    Args:
        docs (list): List of document texts to store.
    Returns:
        collection: The new ChromaDB collection with uploaded docs.
    """
    client = chromadb.Client()
    # Delete existing collection if it exists
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    # Create fresh collection
    collection = client.create_collection(name="docs")
    ids = [f"doc{i+1}" for i in range(len(docs))]
    if docs:
        collection.add(documents=docs, ids=ids)
    return collection

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load and cache the AI model for text2text-generation.
    Returns:
        pipeline: HuggingFace pipeline for text2text-generation.
    """
    return pipeline("text2text-generation", model="google/flan-t5-small")

ai_model = load_model()

def get_answer(collection, question):
    """
    Search documents and generate an answer using the AI model, minimizing hallucination.
    Args:
        collection: ChromaDB collection to search.
        question (str): User's question.
    Returns:
        tuple: (answer, source_docs)
    """
    # STEP 1: Search for relevant documents in the database
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    # STEP 2: Extract search results
    docs = results["documents"][0]
    distances = results["distances"][0]
    # STEP 3: Check if documents are actually relevant to the question
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents.", []
    # STEP 4: Create structured context for the AI model
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    # STEP 5: Build improved prompt to reduce hallucination
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with \"I don't know.\" Do not add information from outside the context.

Answer:"""
    # STEP 6: Generate answer with anti-hallucination parameters
    response = ai_model(
        prompt,
        max_length=150
    )
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    return answer, docs


# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto QA Hub", page_icon="🪙", layout="wide")

# --- SIDEBAR TIPS & INFO ---
with st.sidebar:
    st.markdown("### 🧠 Tips")
    st.markdown("- Ask concise questions\n- Use keywords like Bitcoin, mining, gas fees\n- Clear docs if app misbehaves")
    st.markdown("---")
    st.markdown("Built with ❤️ by Maj • [GitHub](https://github.com/yourusername/blockchain-crypto-qa)")




st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap');
body {
    background: url('https://images.unsplash.com/photo-1621413574681-1b5b8b8b8b8b?auto=format&fit=crop&w=1920&q=80') no-repeat center center fixed;
    background-size: cover;
}
.stApp {
    background-color: rgba(0, 0, 0, 0.78); /* dark overlay to ensure readability */
    font-family: 'Space Grotesk', 'Roboto Mono', monospace, sans-serif;
    color: #e0e0e0 !important;
}
section.main > div.block-container {
    background: rgba(27, 31, 35, 0.96);
    padding: 2.5rem 3.5rem;
    border-radius: 20px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.7);
    border: 2px solid #00e676;
    max-width: 900px;
    margin: auto;
}
.stButton>button {
    background: linear-gradient(45deg, #00e676, #00bfae);
    color: #232526;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 230, 118, 0.3);
}
.stButton>button:hover {
    background: linear-gradient(45deg, #00bfae, #00e676);
    box-shadow: 0 6px 20px rgba(0, 230, 118, 0.5);
    transform: scale(1.07);
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
    background: #232526 !important;
    color: #00e676 !important;
    border: 2px solid #00e676 !important;
    border-radius: 10px !important;
    font-weight: 600;
}
h1, h2, h3, h4 {
    text-shadow: 1px 1px 8px #00e676;
    font-weight: 700;
}
.result-card {
    border: 2px solid #00e676;
    background: rgba(0,0,0,0.5);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 5px 15px rgba(0, 230, 118, 0.15);
    transition: background 0.3s ease;
}
.result-card:hover {
    background: rgba(0, 230, 118, 0.18);
}
.source-card {
    background: rgba(255,255,255,0.03);
    border-left: 4px solid #00e676;
    padding: 10px;
    margin-top: 1rem;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.95rem;
    font-style: italic;
    color: #b2ffec;
}
.crypto-divider {
    height: 30px;
    background: url('https://cryptologos.cc/logos/bitcoin-btc-logo.svg?v=029') repeat-x;
    background-size: contain;
    animation: crypto-wave 15s linear infinite;
    margin: 30px 0;
}
@keyframes crypto-wave {
    0% { background-position-x: 0; }
    100% { background-position-x: 1000px; }
}
.sidebar-section {
    background: rgba(24,28,31,0.92);
    color: #f7c873;
    border-radius: 14px;
    padding: 1.3rem 1.1rem;
    margin-bottom: 1.7rem;
    border: 2px solid #f7c873;
    box-shadow: 0 2px 12px rgba(247,200,115,0.08);
    font-size: 1.08rem;
}
.footer {
    font-size: 0.9rem;
    color: #b2ffec;
    margin-top: 50px;
    padding-top: 30px;
    border-top: 1px solid #00e676;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# --- HEADER & INTRO ---
st.markdown("""
<div style='max-width:900px;margin:auto;'>
<h1 style='text-align:center; color:#00e676; text-shadow: 1px 1px 8px #232526; letter-spacing:1px; font-family:Space Grotesk,sans-serif;'>🪙 Crypto & Blockchain Q&A Hub 🚀</h1>
<div style='text-align:center; font-size:1.25rem; color:#b2ffec; margin-bottom:1.5rem; font-weight:600;'>
Welcome to the <b>Blockchain & Cryptocurrency Knowledge Base</b>! 💡<br>
Ask anything about blockchain technology, Bitcoin, Ethereum, wallets, mining, smart contracts, and more.<br>
Explore the basics and the future of digital money! 🪙🔒
</div>
</div>
""", unsafe_allow_html=True)

# --- WAVE DIVIDER ---
st.markdown('<div class="crypto-divider"></div>', unsafe_allow_html=True)


# --- UNIQUE TOP BANNER (REPLACES SIDEBAR) ---
st.markdown("""
<div style='background:rgba(33,37,41,0.92); border:2px solid #f7c873; border-radius:18px; margin-bottom:2rem; padding:1.2rem 2rem; box-shadow:0 2px 12px rgba(247,200,115,0.10); display:flex; align-items:center; justify-content:space-between;'>
  <div style='display:flex; align-items:center;'>
    <img src='https://img.icons8.com/color/48/000000/bitcoin--v1.png' alt='Bitcoin Logo' width='48' height='48' style='margin-right:18px;'/>
    <span style='font-size:1.25rem; color:#f7c873; font-weight:700;'>Did you know?</span>
  </div>
  <div style='color:#b2ffec; font-size:1.08rem; text-align:left;'>
    <ul style='margin:0; padding-left:1.2em; list-style:none;'>
      <li style='margin-bottom:8px; display:flex; align-items:center;'><img src="https://img.icons8.com/color/22/000000/bitcoin--v1.png" alt="Bitcoin Logo" width="22" height="22" style="margin-right:8px;"/>Bitcoin's supply is capped at 21 million coins.</li>
      <li style='margin-bottom:8px; display:flex; align-items:center;'><img src="https://img.icons8.com/color/22/000000/ethereum.png" alt="Ethereum Logo" width="22" height="22" style="margin-right:8px;"/>Ethereum enables smart contracts and NFTs.</li>
      <li style='margin-bottom:8px; display:flex; align-items:center;'><img src="https://img.icons8.com/color/22/000000/lock--v1.png" alt="Lock Icon" width="22" height="22" style="margin-right:8px;"/>Lost your private key? Your crypto is gone forever!</li>
      <li style='margin-bottom:8px; display:flex; align-items:center;'><img src="https://img.icons8.com/color/22/000000/leaf.png" alt="Leaf Icon" width="22" height="22" style="margin-right:8px;"/>Proof of Stake is more eco-friendly than mining.</li>
      <li style='margin-bottom:8px; display:flex; align-items:center;'><img src="https://img.icons8.com/color/22/000000/anonymous-mask.png" alt="Anonymous Icon" width="22" height="22" style="margin-right:8px;"/>Crypto transactions are public, but your name isn't.</li>
    </ul>
  </div>
  <div style='display:flex; align-items:center;'>
    <img src='https://img.icons8.com/color/48/000000/ethereum.png' alt='Ethereum Logo' width='48' height='48' style='margin-left:18px;'/>
  </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
st.markdown("<div style='max-width:900px;margin:auto;'>", unsafe_allow_html=True)

st.markdown("## 📄 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload documents to add to your knowledge base (PDF, DOC, DOCX, TXT)",
    type=["pdf", "doc", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload no more than 5 documents at once.")
    else:
        st.write(f"You selected {len(uploaded_files)} file(s). Converting...")
        converted_docs = convert_and_store(uploaded_files)
        st.session_state.converted_docs = converted_docs
        st.session_state.collection = setup_collection_from_upload(converted_docs)
        for i, text in enumerate(converted_docs):
            st.success(f"File {i+1} converted! Preview:")
            st.text(text[:500] + ("..." if len(text) > 500 else ""))
        st.success("Documents successfully processed and stored.")
with st.expander("📄 Document Manager"):
    if 'converted_docs' in st.session_state and st.session_state.converted_docs:
        st.write("Uploaded documents:")
        for i, file in enumerate(uploaded_files or []):
            st.markdown(f"- {file.name}")
        if st.button("🗑️ Clear All Documents"):
            st.session_state.converted_docs = []
            st.session_state.collection = None
            st.session_state.search_history = []
            st.experimental_rerun()
    else:
        st.info("No documents uploaded yet.")

st.markdown("## ❓ Ask Your Question")
question = st.text_area("💬 Type your question", height=100, placeholder="E.g. How does mining work?")
st.caption("💡 Example: What is Proof of Stake and how does it save energy?")

st.markdown("## 💬 Answer & Sources")
if st.button("🪙 Get My Crypto Answer", type="primary"):
    if not st.session_state.collection:
        st.warning("⚠️ Please upload at least one document to build your knowledge base before asking a question.")
    elif not question:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🧠 Crunching crypto ledgers... ⛓️"):
            answer, sources = get_answer(st.session_state.collection, question)
        st.session_state.search_history.append(question)
        st.markdown(
            f'''<div style="border: 2px solid #00e676; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 15px; margin-bottom: 1rem;">
            <h4 style="color: #00e676;">✅ AI Response</h4>
            <p style="color: #e0e0e0;">{answer}</p>
            </div>''', unsafe_allow_html=True)
        st.markdown("**Sources:**")
        for i, source in enumerate(sources):
            st.markdown(f'''<div class="source-card"><code>Doc {i+1}: {source[:200]}...</code></div>''', unsafe_allow_html=True)
        # --- Feedback Loop ---
        st.markdown("### 👍 Was this answer helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎 No"):
                st.info("We’ll try to improve next time.")

with st.expander("🕘 Search History"):
    if st.session_state.search_history:
        for i, q in enumerate(st.session_state.search_history[::-1], 1):
            st.markdown(f"{i}. {q}")
        if st.button("Export Q&A History as TXT"):
            output = io.StringIO()
            for i, q in enumerate(st.session_state.search_history, 1):
                output.write(f"Q{i}: {q}\n")
            st.download_button(
                label="Download Q&A History",
                data=output.getvalue(),
                file_name="qa_history.txt",
                mime="text/plain"
            )
    else:
        st.info("No questions asked yet.")

st.markdown('<div class="crypto-divider"></div>', unsafe_allow_html=True)

with st.expander("📘 How to Use This App"):
    st.markdown("""
    - Upload up to 5 documents (PDF, DOCX, TXT)
    - Ask clear and specific questions
    - View search history below
    - Use the document manager to reset or review
    """)

st.markdown("</div>", unsafe_allow_html=True)


# --- Modern Footer ---
st.markdown("""
<div class="footer">
Built with ❤️ by Maj • 2025 <br>
<a href='https://github.com/yourusername/blockchain-crypto-qa' target='_blank'>View Source on GitHub</a>
</div>
""", unsafe_allow_html=True)

# TO RUN: Save as app.py, then type: streamlit run app.py
##
# STREAMLIT BUILDING BLOCKS SUMMARY:
# =================================
# 1. st.title(text) - Creates the main heading of your app
# 2. st.write(text) - Displays text, data, or markdown content
# 3. st.text_input(label, placeholder="hint") - Creates a text box for user input
# 4. st.button(text, type="primary") - Creates a clickable button
# 5. st.spinner(text) - Shows a spinning animation with custom text
# 6. st.expander(title) - Creates a collapsible section
# HOW THE APP FLOW WORKS:
# 1. User opens browser → Streamlit loads the app
# 2. setup_documents() runs → Creates document database
# 3. st.title() and st.write() → Display app header
# 4. st.text_input() → Shows input box for questions
# 5. st.button() → Shows the "Get Answer" button
# 6. User types question and clicks button:
#    - if statement triggers
#    - st.spinner() shows loading animation
#    - get_answer() function runs in background
#    - st.write() displays the result
# 7. st.expander() → Shows help section at bottom
# WHAT HAPPENS WHEN USER INTERACTS:
# - Type in text box → question variable updates automatically
# - Click button → if st.button() becomes True
# - Spinner shows → get_answer() function runs
# - Answer appears → st.write() displays the result
# - Click expander → help section shows/hides
# This creates a simple but complete web application!
