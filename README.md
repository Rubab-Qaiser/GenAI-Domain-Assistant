# GenAI Domain Assistant Chatbot

A domain-specific chatbot that answers domain based questions using Retrieval-Augmented Generation (RAG) with Google Gemini.  
It only answers based on the documents you provide, reducing hallucinations and keeping responses grounded in your data.

Built with Python, LangChain, and Google Gemini 2.5 Flash.

## Features

- **Document Ingestion**: Load PDFs and text files containing HR policies.
- **Text Chunking**: Split documents into optimal chunks with configurable size and overlap.
- **RAG Pipeline**: Retrieve relevant context and generate grounded answers.
- **Keyword Retrieval**: Lightweight baseline retriever for testing without embeddings.
- **Comparison Mode**: Run queries with and without RAG to see the difference.
- **Debugging**: Print retrieved chunks to verify what the model sees.

## Project Structure
genai-hr-assistant/
│
├── data/                    # Place your HR policy PDFs/txt files here
├── notebooks/
│   └── rag_hr_assistant.ipynb  # Main notebook with full pipeline
├── README.md
## requirements.txt
javascriptgoogle-generativeai
langchain
langchain-community
pypdf
kaggle_secrets

## Add your Google Gemini API Key
This project uses Kaggle Secrets for key management. 
In Kaggle Notebooks:Go to Add-ons > Secrets
Add a secret named GEMINI_API_KEY with your API key from Google AI StudioLocally, you can use os.environ["GEMINI_API_KEY"] = "your_key"
##  Add your documents
Place your HR policy PDFs or text files in the data/ folder.
## Limitations:
Keyword retrieval only: The current simple_search uses exact keyword matching. 
It won’t match synonyms like "parental leave" vs "maternity leave".
Limited by source data: The bot cannot answer questions not covered in your documents. 
This is intentional to prevent hallucination.No conversation memory: Each query is independent. 
Add chat history if you need follow-up questions.
## Next Steps:
1. To improve the system:Replace simple_search with FAISS + HuggingFace embeddings for semantic search.
2. Add more HR policy documents to expand coverage.Implement chat history for multi-turn conversations.
3. Deploy as a Streamlit or FastAPI app.
