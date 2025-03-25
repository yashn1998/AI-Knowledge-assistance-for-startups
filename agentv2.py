import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredXMLLoader
from langchain.prompts import PromptTemplate
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables. Please set it in .env file.")

print("Imports and logging setup complete.")


# Initialize LLM and Embedding Model
llm = ChatGroq(
    temperature=0.8,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    max_tokens=2000
)
# embedder = HuggingFaceEmbeddings(
#     model_name="mixedbread-ai/mxbai-embed-large-v1",
#     model_kwargs={"device": "cpu"}
# )

print("LLM and Embeddings initialized.")

# State Definition
class ChatbotState(TypedDict):
    query: str
    retrieved_docs: List[str]
    response: str
    metadata: Dict[str, Any]
    prompt_type: str
    conversation_history: List[Dict[str, str]]
    requires_human_review: bool

# Prompt Library with Advanced Techniques
class PromptLibrary:
    def __init__(self):
        self.prompts = {
            "qa": {
                "template": """Answer '{question}' using this context: {context}
- If the context fully answers the question, provide a concise, accurate response with key details highlighted.
- If the context is partial, give a brief answer based on available info and note what’s missing.
- If no relevant info is found, say: "The document doesn’t provide enough information to answer this."
Focus on clarity, relevance, and actionable insights.""",
                "version": "1.0",
                "tags": ["qa", "least-to-most"],
                "description": "Breaks down questions systematically."
            },
            "summarization": {
                "template": """Summarize this document content: {context}
- Identify and prioritize key sections (e.g., personal info, skills, experience, projects).
- Provide a concise, structured summary (e.g., bullet points or short paragraphs) that captures the essence.
- Exclude irrelevant details and focus on what makes the content unique or valuable.
Aim for clarity, brevity, and a professional tone.""",
                "version": "1.1",
                "tags": ["summarization", "self-refinement"],
                "description": "Self-refines for brevity."
            },
            "creative": {
                "template": """Generate a creative response to '{question}' using context: {context}
Focus on inspiration and positivity.""",
                "version": "1.0",
                "tags": ["creative", "directional-stimulus"],
                "description": "Produces uplifting responses."
            }
        }

    def get_prompt(self, prompt_type: str, query: str, context: str) -> PromptTemplate:
        if prompt_type not in self.prompts:
            logger.warning(f"Prompt type '{prompt_type}' not found, defaulting to 'qa'")
            prompt_type = "qa"
        data = self.prompts[prompt_type]
        return PromptTemplate(
            template=data["template"],
            input_variables=["context", "question"] if "question" in data["template"] else ["context"],
            metadata={"version": data["version"], "tags": data["tags"]}
        )

        
    def route_prompt(self, query: str) -> str:
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Keyword-based intent detection
            # Fallback to LLM-based classification if keywords don't match
        messages = [
            SystemMessage(content="If user is asking question about documnet then classisfy intent as 'qa' . If user is asking about to summarize then do 'summarization' .Classify the intent as 'qa', 'summarization' only. Provide only the intent name."),
            HumanMessage(content=query)
        ]
        intent = llm.invoke(messages).content.strip().lower()
        return intent if intent in self.prompts else "qa"


prompt_lib = PromptLibrary()
print("State and Prompt Library defined.")




import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredXMLLoader
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment variables. Please set it in .env file.")

# Configure logging with detailed format and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedEmbedder:
    def __init__(self):
        # Using a multilingual model with 768 dimensions
        self.model = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")
        logger.info("Initialized advanced embedder: stsb-xlm-r-multilingual (768 dimensions)")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        logger.info(f"Embedded {len(texts)} documents in {time.time() - start_time:.2f} seconds")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        start_time = time.time()
        embedding = self.model.encode([query])[0]
        logger.info(f"Embedded query in {time.time() - start_time:.2f} seconds")
        return embedding.tolist()

class RAGSystem:
    def __init__(self, index_name="rag-index", dimension=768, cloud="aws", region="us-east-1"):
        """
        Initialize the RAG system with Pinecone and advanced components.
        
        Args:
            index_name (str): Name of the Pinecone index.
            dimension (int): Dimension of the embedding vectors (768 for stsb-xlm-r-multilingual).
            cloud (str): Cloud provider (e.g., "aws", "gcp", "azure").
            region (str): Region for the cloud provider (e.g., "us-east-1" for AWS).
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = AdvancedEmbedder()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Initialized once
        logger.info("Initialized CrossEncoder: ms-marco-MiniLM-L-6-v2 for reranking")
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = None
        self._initialize_pinecone()

    def _initialize_pinecone(self) -> None:
        """Initialize or connect to a Pinecone index with region validation."""
        supported_regions = {
            "aws": ["us-east-1", "us-west-2", "eu-west-1"],
            "gcp": ["us-central1", "europe-west1"],
            "azure": ["eastus"]
        }
        if self.cloud not in supported_regions:
            raise ValueError(f"Unsupported cloud provider: {self.cloud}. Supported: {list(supported_regions.keys())}")
        if self.region not in supported_regions[self.cloud]:
            raise ValueError(f"Region {self.region} not supported for {self.cloud}. Supported: {supported_regions[self.cloud]}")

        if self.index_name not in self.pc.list_indexes().names():
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region)
                )
                logger.info(f"Created new Pinecone index: {self.index_name} with dimension {self.dimension}")
                time.sleep(5)  # Wait for index creation to propagate
            except Exception as e:
                logger.error(f"Failed to create index {self.index_name}: {str(e)}")
                raise
        else:
            logger.info(f"Index {self.index_name} already exists.")

        self.index = self.pc.Index(self.index_name)
        logger.info(f"Connected to Pinecone index: {self.index_name}")

    def _parallel_upsert(self, vectors: List[dict], namespace: str, batch_size=100) -> None:
        """Perform parallel upsert for faster indexing with detailed logging."""
        def upsert_batch(batch):
            start_time = time.time()
            self.index.upsert(vectors=batch, namespace=namespace)
            logger.info(
                f"Upserted batch of {len(batch)} vectors in {time.time() - start_time:.2f} seconds "
                f"(namespace: {namespace}, IDs: {[v['id'] for v in batch]})"
            )

        total_chunks = len(vectors)
        logger.info(f"Starting parallel upsert of {total_chunks} vectors into namespace: {namespace}")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                executor.submit(upsert_batch, batch)

        logger.info(f"Completed upsert of {total_chunks} vectors in {time.time() - start_time:.2f} seconds")
        time.sleep(2)  # Wait for upsert to propagate

    def load_document(self, file_path: str, namespace: str = "default") -> None:
        """Load and process a document with advanced strategies and detailed logging."""
        file_path = os.path.normpath(file_path.strip())
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".xml": UnstructuredXMLLoader
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

        start_time = time.time()
        loader = loaders[ext](file_path)
        docs = loader.load()
        logger.info(f"Loaded document: {file_path} in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        chunks = self.text_splitter.split_documents(docs)
        chunk_texts = [doc.page_content for doc in chunks]
        logger.info(f"Split document into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")

        embeddings = self.embeddings.embed_documents(chunk_texts)
        vectors = []
        start_time = time.time()
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            vector_id = f"{file_path}_{i}"
            vectors.append({"id": vector_id, "values": embedding, "metadata": {"text": text}})
            if i < 3:
                logger.info(
                    f"Created vector {i+1}/{len(chunk_texts)}: ID={vector_id}, "
                    f"Embedding length={len(embedding)}, Sample text={text[:50]}..."
                )
        logger.info(f"Generated {len(vectors)} vectors in {time.time() - start_time:.2f} seconds")

        self._parallel_upsert(vectors, namespace)
        logger.info(f"Completed loading and upserting {file_path} into namespace: {namespace}")

    def retrieve(self, query: str, k: int = 6, namespace: str = "default", initial_k: int = 20) -> List[str]:
        """Retrieve relevant document chunks using advanced techniques like query expansion and reranking."""
        if not self.index:
            logger.warning("No Pinecone index initialized.")
            return []

        start_time = time.time()

        # Step 1: Query Expansion - Generate a richer query
        expanded_query = self._expand_query(query)
        logger.info(f"Expanded query: {expanded_query}")

        # Step 2: Initial Retrieval - Fetch more candidates than needed for reranking
        query_embedding = self.embeddings.embed_query(expanded_query)
        initial_result = self.index.query(
            vector=query_embedding,
            top_k=initial_k,
            include_metadata=True,
            namespace=namespace
        )
        initial_docs = [
            {"text": match["metadata"]["text"], "score": match["score"]}
            for match in initial_result["matches"]
        ]
        logger.info(f"Initially retrieved {len(initial_docs)} chunks in {time.time() - start_time:.2f} seconds")

        if not initial_docs:
            logger.info("No documents retrieved from initial query.")
            return []

        # Step 3: Reranking - Use pre-initialized CrossEncoder
        reranked_docs = self._rerank_docs(query, initial_docs)
        
        # Step 4: Select top-k reranked documents
        final_docs = [doc["text"] for doc in reranked_docs[:k]]
        logger.info(f"Reranked and selected top {len(final_docs)} chunks in {time.time() - start_time:.2f} seconds")

        return final_docs

    def _expand_query(self, query: str) -> str:
        """Expand the query using synonyms or context to improve retrieval recall."""
        expansion_terms = {
            "project": "work experience task",
            "skill": "ability expertise",
            "summary": "overview brief",
            "what": "details information"
        }
        query_lower = query.lower()
        expanded = query
        for term, synonyms in expansion_terms.items():
            if term in query_lower:
                expanded += f" {synonyms}"
                break  # Simple: expand only once for the first match
        
        return expanded.strip()

    def _rerank_docs(self, query: str, docs: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Rerank retrieved documents using the pre-initialized CrossEncoder."""
        try:
            logger.info("Reranking with pre-initialized CrossEncoder")
            pairs = [[query, doc["text"]] for doc in docs]
            scores = self.reranker.predict(pairs)
            for i, doc in enumerate(docs):
                doc["rerank_score"] = float(scores[i])
            reranked_docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
            logger.info(f"Reranked {len(reranked_docs)} documents")
            return reranked_docs
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}. Falling back to initial ranking.")
            return sorted(docs, key=lambda x: x["score"], reverse=True)

    def clear(self, namespace: str = "default") -> None:
        """Clear a namespace in the Pinecone index."""
        if self.index:
            start_time = time.time()
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Cleared Pinecone namespace: {namespace} in {time.time() - start_time:.2f} seconds")
        else:
            logger.warning("No Pinecone index to clear.")


rag = RAGSystem(index_name="rag-index", dimension=768, cloud="aws", region="us-east-1")
print("RAG System with Pinecone initialized (768 dimensions).")





def intent_agent(state: ChatbotState) -> ChatbotState:
    state['query'] = state['query'].strip()
    if len(state['query']) < 5 or not any(c.isalpha() for c in state['query']):  # Check for short or non-alphabetic input
        state['response'] = "Sorry, your input is unclear. Could you please provide more details?"
        state['requires_human_review'] = False
        logger.info("Ambiguous input detected, requesting clarification")
        return state
    state['prompt_type'] = prompt_lib.route_prompt(state['query'])
    state['conversation_history'].append({"query": state['query'], "prompt_type": state['prompt_type'], "response": ""})
    logger.info(f"Intent detected: {state['prompt_type']}")
    return state

    

def retrieval_agent(state: ChatbotState) -> ChatbotState:
    """Retrieves and refines relevant document chunks based on the query."""
    # Step 1: Retrieve initial documents
    initial_docs = rag.retrieve(state['query'])  # Increase to 5 for broader coverage
    logger.info(f"Initially retrieved {len(initial_docs)} chunks")

    if not initial_docs:
        state['retrieved_docs'] = []
        state['response'] = "No relevant content found in the loaded document."
        logger.info("No documents retrieved")
        return state

    # Step 2: Use LLM to refine and filter retrieved chunks
    context = "\n\n".join(initial_docs)
    refinement_prompt = f"""Given the query: '{state['query']}', refine this context into a concise version:
                            {context}
                            - Keep only the parts directly relevant to the query.
                            - Remove redundant or off-topic information.
                            - Return the refined text or 'Insufficient relevant content' if nothing matches."""
    
    messages = [
        SystemMessage(content="You are an expert at refining text. Focus strictly on relevance to the query."),
        HumanMessage(content=refinement_prompt)
    ]
    
    try:
        refined_response = llm.invoke(messages).content.strip()
        if "insufficient relevant content" in refined_response.lower():
            state['retrieved_docs'] = []
            state['response'] = "The document doesn’t contain enough relevant information for your query."
            logger.info("Refined retrieval: No relevant content found")
        else:
            state['retrieved_docs'] = [refined_response]  # Store as a single refined chunk
            logger.info(f"Refined {len(initial_docs)} chunks into 1 query-specific chunk")
    except Exception as e:
        state['retrieved_docs'] = initial_docs  # Fallback to unrefined docs
        logger.error(f"Retrieval refinement failed: {str(e)}")
    
    return state

def qa_agent(state: ChatbotState) -> ChatbotState:
    if not state['retrieved_docs']:
        state['response'] = "No document loaded or no relevant content found."
    else:
        context = "\n".join(state['retrieved_docs'])
        prompt = prompt_lib.get_prompt("qa", state['query'], context)
        messages = [
            SystemMessage(content="Your are Knowledge asistance for startups. Answer questions accurately using only the provided context. If the context lacks sufficient information, say so."),
            HumanMessage(content=prompt.format(context=context, question=state['query']))
        ]
        state['response'] = llm.invoke(messages).content
        if "no relevant content" in state['response'].lower():
            state['response'] = "The document doesn’t provide enough information to answer this question."
    logger.info("QA response generated")
    return state







def guardrail_agent(state: ChatbotState) -> ChatbotState:
    # Safety check
    messages = [
        SystemMessage(content="Evaluate this response: {response}. if everything looks good then only procced. . "),
        HumanMessage(content=state['response'])
    ]
    return state

def summarization_agent(state: ChatbotState) -> ChatbotState:
    """Generates a query-specific summary based on retrieved document content."""
    if not state['retrieved_docs']:
        state['response'] = "No document loaded or no relevant content found to summarize."
        logger.info("No content available for summarization")
        return state

    # Use the refined context from retrieval_agent
    context = "\n".join(state['retrieved_docs'])
    
    # Tailor the summarization prompt based on the query
    prompt = prompt_lib.get_prompt("summarization", state['query'], context)
    messages = [
        SystemMessage(content="You are a summarization expert. Summarize the provided context concisely, focusing on aspects relevant to the query."),
        HumanMessage(content=prompt.format(context=context))
    ]
    
    try:
        summary = llm.invoke(messages).content.strip()
        if "insufficient" in summary.lower() or not summary:
            state['response'] = "The document doesn’t provide enough relevant information for a meaningful summary."
        else:
            # Add structure to the summary based on query intent
            if "experience" in state['query'].lower() or "projects" in state['query'].lower():
                state['response'] = f"Summary of relevant experience/projects:\n{summary}"
            else:
                state['response'] = f"Summary:\n{summary}"
        logger.info("Summarization response generated")
    except Exception as e:
        state['response'] = "Sorry, I couldn’t summarize the content due to an issue."
        state['requires_human_review'] = True
        logger.error(f"Summarization failed: {str(e)}")
    
    return state
    




print("Workflow nodes defined.")



def build_workflow():
    workflow = StateGraph(ChatbotState)
    # Define nodes
    workflow.add_node("intent", intent_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("summarization", summarization_agent)
    workflow.add_node("qa", qa_agent)
    workflow.add_node("guardrail", guardrail_agent)
    
    # Set entry point
    workflow.set_entry_point("intent")
    
    # From intent to retrieval for all cases
    workflow.add_edge("intent", "retrieval")
    
    # Conditional routing after retrieval based on intent
    workflow.add_conditional_edges(
        "retrieval",
        lambda state: state['prompt_type'],
        {
            "qa": "qa",
            "summarization": "summarization"
        }
    )
    
    # Edges to guardrail (last step before end)
    workflow.add_edge("qa", "guardrail")
    workflow.add_edge("summarization", "guardrail")
    
    # Guardrail to end
    workflow.add_edge("guardrail", END)
    
    return workflow.compile()

app = build_workflow()
print("Workflow built.")



from frontend import FastAPI, File, UploadFile, HTTPException, Form
from typing import Optional
import os
import logging
import asyncio
from dotenv import load_dotenv
import nest_asyncio
from langgraph.graph import END
import aiofiles  # For async file operations
import uvicorn
import tempfile

# Import your main backend components (adjust as needed)
# from your_workflow_module import build_workflow, ChatbotState, rag

# Apply nest_asyncio for compatibility (only if needed in your environment)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler("fastapi.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with production settings
app = FastAPI(
    title="AI Knowledge Assistant API",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV") != "production" else None,  # Disable docs in production
    redoc_url=None  # Disable ReDoc
)

# Build workflow at startup (assumed to be async-compatible)
workflow_app = build_workflow()
logger.info("Workflow initialized.")

# Constants
MAX_FILE_SIZE = 5_242_880  # 5 MB
TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# API Endpoints
@app.post("/query")
async def query_without_file(query: str = Form(...), prompt_version: Optional[str] = Form("1.1")):
    """Handle text-only queries asynchronously."""
    try:
        state = ChatbotState(
            query=query,
            retrieved_docs=[],
            response="",
            metadata={"status": "pending", "prompt_version": prompt_version},
            prompt_type="",
            conversation_history=[],
            requires_human_review=False
        )
        logger.info(f"Processing query: {query}")
        result = await workflow_app.ainvoke(state)  # Ensure ainvoke is async
        logger.info("Query processed successfully.")
        return {
            "response": result["response"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/query_with_file")
async def query_with_file(
    query: str = Form(...),
    file: UploadFile = File(...),
    prompt_version: Optional[str] = Form("1.1")
):
    """Handle queries with file uploads asynchronously."""
    try:
        # Read file content asynchronously
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 5 MB limit.")

        # Save uploaded file temporarily using async I/O
        file_path = os.path.join(TEMP_DIR, file.filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        
        # Load document into RAG (assuming rag.load_document is async or can be awaited)
        await asyncio.to_thread(rag.load_document, file_path)  # Wrap in thread if sync
        logger.info(f"Document loaded: {file.filename}")

        # If query is just to load the file, return confirmation
        if query.lower().strip() in ["load file", "upload file", ""]:
            return {
                "response": f"File '{file.filename}' loaded and ready for RAG.",
                "metadata": {"status": "loaded", "prompt_version": prompt_version, "file": file.filename}
            }

        # Process the query with the workflow
        state = ChatbotState(
            query=query,
            retrieved_docs=[],
            response="",
            metadata={"status": "pending", "prompt_version": prompt_version, "file": file.filename},
            prompt_type="",
            conversation_history=[],
            requires_human_review=False
        )
        
        logger.info(f"Processing query with file: {query}")
        result = await workflow_app.ainvoke(state)  # Ensure ainvoke is async
        
        return {
            "response": result["response"],
            "metadata": result["metadata"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing query with file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup: Remove temp file asynchronously
        if os.path.exists(file_path):
            await asyncio.to_thread(os.remove, file_path)  # Wrap in thread if sync

@app.get("/health")
async def health_check():
    return {"status": "healthy", "date": "2025-03-25"}

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FastAPI server...")

# Run the app with production-grade settings
# if __name__ == "__main__":
#     logger.info("Starting FastAPI server...")
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         # workers=int(os.getenv("UVICORN_WORKERS", 4)),  # Adjust workers for production
#         log_level="info",
#         timeout_keep_alive=1000  # Handle long-lived connections
#     )
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),  # Use Render's PORT or default to 8000 locally
        log_level="info",
        timeout_keep_alive=1000
    )