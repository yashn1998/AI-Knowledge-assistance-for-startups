
# AI-Knowledge-assistance-for-startups
This is the AI assistance for knowledge.
![image](https://github.com/user-attachments/assets/4be3fde6-102e-4fbe-9f98-a64e5cc2b563)



Below is a detailed explanation of your project that you can use in a GitHub repository README file. It provides an overview, setup instructions, features, and usage details tailored to your codebase.

---

# AI Knowledge Assistant

**A scalable, Retrieval-Augmented Generation (RAG) system built with LangChain, Pinecone, and FastAPI.**

This project is an AI-powered knowledge assistant designed to process queries with or without uploaded documents (e.g., PDFs, DOCX, TXT, XML). It leverages advanced natural language processing (NLP) techniques, vector search, and a modular workflow to provide accurate and context-aware responses. The system integrates a Retrieval-Augmented Generation (RAG) pipeline with Pinecone for vector storage, LangChain for LLM interactions, and FastAPI for a production-ready API.

The assistant supports intent detection (e.g., question-answering or summarization), document retrieval, and response generation, all while ensuring scalability and extensibility. It’s built with startup use cases in mind, such as analyzing documents for insights or summarizing key details.

---

## Features

- **Intent Detection**: Automatically classifies user queries as "qa" (question-answering) or "summarization" using a lightweight LLM-based classifier.
- **Document Processing**: Supports multiple file formats (PDF, DOCX, TXT, XML) with chunking and embedding for retrieval.
- **RAG Pipeline**: Combines Pinecone vector search with LLM refinement and reranking for relevant document retrieval.
- **Modular Workflow**: Built with LangGraph for a stateful, multi-agent architecture (intent, retrieval, QA, summarization, guardrails).
- **API Interface**: FastAPI-based endpoints for querying with or without file uploads, including async support and health checks.
- **Scalability**: Parallel upsert to Pinecone and configurable cloud deployment (AWS, GCP, Azure).
- **Guardrails**: Ensures safe and relevant responses with a final review step.
- **Logging**: Detailed logging for debugging and monitoring.

---

## Tech Stack

- **Python**: Core language (3.9+ recommended).
- **LangChain**: LLM orchestration and prompt management.
- **Pinecone**: Vector database for document embeddings.
- **SentenceTransformers**: Advanced embeddings (multilingual, 768 dimensions).
- **CrossEncoder**: Reranking retrieved documents for relevance.
- **FastAPI**: Asynchronous API framework.
- **Grok (xAI)**: Hypothetical LLM integration (replaceable with any LangChain-compatible model).
- **Uvicorn**: Production-grade ASGI server.

---

## Project Structure

```
├── frontend.py          # FastAPI app with API endpoints
├── rag_system.py        # RAG pipeline with Pinecone and embeddings
├── workflow.py          # LangGraph workflow and agents
├── prompt_library.py    # Custom prompt templates
├── requirements.txt     # Dependencies
├── .env.example         # Sample environment variables
├── rag_system.log       # Log file (generated)
└── README.md            # This file
```

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- API keys for:
  - Grok (xAI) or alternative LLM provider (e.g., OpenAI, Hugging Face).
  - Pinecone (vector database).
- Git (for cloning the repository).

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ai-knowledge-assistant.git
   cd ai-knowledge-assistant
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` with your API keys:
     ```
     GROQ_API_KEY=your-grok-api-key
     PINECONE_API_KEY=your-pinecone-api-key
     PORT=8000  # Optional: defaults to 8000
     ```

5. **Run the Application**:
   ```bash
   python frontend.py
   ```
   The API will be available at `http://localhost:8000`.

---

## Usage

### API Endpoints

1. **Health Check**:
   - **GET `/health`**:
     - Returns: `{"status": "healthy", "date": "2025-03-25"}`

2. **Query Without File**:
   - **POST `/query`**:
     - Body: `query` (string), `prompt_version` (optional, default: "1.1")
     - Example:
       ```bash
       curl -X POST "http://localhost:8000/query" -F "query=What is the project about?"
       ```
     - Response: JSON with `response` and `metadata`.

3. **Query With File**:
   - **POST `/query_with_file`**:
     - Body: `query` (string), `file` (file upload), `prompt_version` (optional)
     - Example:
       ```bash
       curl -X POST "http://localhost:8000/query_with_file" \
            -F "query=Summarize this document" \
            -F "file=@path/to/yourfile.pdf"
       ```
     - Response: JSON with `response` and `metadata`.

### Example Queries
- **QA**: "What skills are mentioned in the document?"
- **Summarization**: "Summarize the experience section."

---

## How It Works

1. **User Input**: Submit a query via the API, optionally with a file.
2. **File Processing**: Uploaded files are chunked, embedded, and stored in Pinecone.
3. **Intent Detection**: The system identifies if the query is for QA or summarization.
4. **Retrieval**: Relevant document chunks are fetched and refined using Pinecone and CrossEncoder.
5. **Response Generation**: The LLM generates a response based on the intent (QA or summary).
6. **Guardrails**: Ensures the response is safe and relevant.
7. **Output**: Returns the result via the API.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Future Improvements

- Add support for more file types (e.g., images with OCR).
- Implement conversation history persistence.
- Optimize embedding and retrieval for larger datasets.
- Deploy to a cloud provider (e.g., AWS, Render) with CI/CD.

---

Feel free to customize this further based on your specific needs (e.g., adding a deployment section for Render, updating the repository URL, or refining the features list). Let me know if you'd like help with any specific part!
