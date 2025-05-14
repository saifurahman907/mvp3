from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import shutil
import threading
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# LangChain imports - grouped by module
# Core components
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# OpenAI components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Document processing 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

# Vector stores
from langchain_chroma import Chroma

# Utility imports
from operator import itemgetter

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly and fail fast if not
if not api_key:
    logger.critical("API key not found. Please check your .env file.")
    raise EnvironmentError("OpenAI API key not found in environment variables")
else:
    logger.info("API key loaded successfully")

# Flask App Setup with properly configured CORS
app = Flask(__name__)

# IMPROVED CORS CONFIGURATION
# Default to all origins if not specified - better for development
DEFAULT_ORIGINS = "*"
allowed_origins = os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS)
if allowed_origins == "*":
    # Enable CORS for all origins with support for credentials
    CORS(app, supports_credentials=True)
    logger.info("CORS enabled for all origins")
else:
    # Enable CORS for specific origins
    origins = allowed_origins.split(",")
    CORS(app, 
         resources={r"/*": {"origins": origins}},
         supports_credentials=True,
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"])
    logger.info(f"CORS enabled for specific origins: {origins}")

# Configuration parameters that can be set from environment
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "5"))
SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
HOST = os.getenv("HOST", "127.0.0.1")  # Default to localhost for security
PORT = int(os.getenv("PORT", "5000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "20"))

# Base directory for vector stores - using Path for better path handling
VECTOR_STORE_BASE_DIR = Path("vector_stores")
VECTOR_STORE_BASE_DIR.mkdir(exist_ok=True)

# Temporary upload directory
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

# Thread-safe dictionaries using locks
class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
        
    def __contains__(self, key):
        with self._lock:
            return key in self._dict
            
    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value
            
    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]
            
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
            
    def items(self):
        with self._lock:
            return list(self._dict.items())
            
    def keys(self):
        with self._lock:
            return list(self._dict.keys())
            
    def clear(self):
        with self._lock:
            self._dict.clear()

# Thread-safe collections for shared state
contract_vector_stores = ThreadSafeDict()  # contract_id -> vector_store_path
processing_status = ThreadSafeDict()       # contract_id -> status
message_histories = ThreadSafeDict()       # contract_id -> ChatMessageHistory
session_creation_times = ThreadSafeDict()  # contract_id -> datetime

# Cache for Chroma instances
chroma_cache = ThreadSafeDict()  # contract_id -> Chroma instance

# Initialize embeddings and text splitter
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Create a thread pool for background processing
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

# Context manager for Chroma to ensure proper cleanup
@contextmanager
def get_chroma_db(contract_id):
    """Context manager to get a Chroma instance and ensure proper cleanup"""
    # Check if we have a cached instance
    db = None
    create_new = False
    
    if contract_id in chroma_cache:
        db = chroma_cache[contract_id]
    else:
        # Need to create a new connection
        if contract_id not in contract_vector_stores:
            logger.error(f"No vector store found for contract ID: {contract_id}")
            yield None
            return
            
        vector_store_path = contract_vector_stores[contract_id]
        
        # Check if the directory exists
        if not Path(vector_store_path).exists():
            logger.error(f"Vector store directory does not exist: {vector_store_path}")
            # Remove from tracking if directory doesn't exist
            if contract_id in contract_vector_stores:
                del contract_vector_stores[contract_id]
            yield None
            return
        
        # Initialize the vector store for this contract
        try:
            db = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding
            )
            create_new = True
        except Exception as e:
            logger.error(f"Error initializing Chroma for contract {contract_id}: {str(e)}")
            if contract_id in contract_vector_stores:
                del contract_vector_stores[contract_id]
            yield None
            return
    
    try:
        yield db
    finally:
        # If we created a new connection, cache it for reuse
        if create_new and db:
            chroma_cache[contract_id] = db

def format_docs(docs):
    """Format documents into a single string"""
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever(contract_id):
    """Get a retriever for the specified contract"""
    try:
        with get_chroma_db(contract_id) as vectordb:
            if not vectordb:
                return None
                
            # Check if there are documents in the vector store
            docs_info = vectordb.get()
            if len(docs_info["documents"]) == 0:
                logger.warning(f"No documents in vector store for contract ID: {contract_id}")
                return None
                
            return vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    except Exception as e:
        logger.error(f"Error getting retriever for contract {contract_id}: {str(e)}")
        return None

# LLM with timeout configuration
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=api_key,
    request_timeout=90,
    max_retries=3
)

def load_prompt_from_file(filename):
    """Load prompt template from file"""
    try:
        with open(filename, 'r') as file:
            return file.read()
    except FileNotFoundError:
        # If file doesn't exist, return the default prompt
        if filename == "prompts/summary_prompt.txt":
            return """
            History:
            {history}

            Context:
            {context}

            Role: You are a contract law expert specializing in UK construction contracts. Your audience is non-technical construction professionals who are not experts in contracts.

            Task: Using the provided context, produce a detailed summary of the contract. Write in clear, simple, everyday language, and explain any technical terms so that a layperson can easily understand.

            FORMAT REQUIREMENTS:
            - Use CAPITALIZED HEADINGS for main sections (e.g., DOCUMENTS, PAYMENTS, RETENTION).
            - For each section, include an **Indented Subheading**: the prompt question from the left-hand column of the prompt table.
            - Provide **Bullet points** with answers extracted from the contract to the prompt question.
            - Keep explanations brief and in layman's terms.
            - Provide **reasonable context** for each answer.
            - Use **bold** markdown for key dates, values, and timeframes (e.g., **35 days**, **March 2022**, **$10,000**).
            - All responses should be clear and natural for non-technical construction professionals, making it easy for them to understand.

            Please address the key contract areas focusing on:
            1. DOCUMENTS
            2. CONTRACT FORM
            3. PAYMENTS
            4. TERMINATION
            5. SUSPENSION
            6. VARIATIONS
            7. KEY RISKS

            Questions:
            {question}

            - Ensure that each section starts with the heading in **CAPITALIZED** format.
            - Ensure that each **subheading (question)** is clearly indented.
            - Each answer should be **bullet-pointed** for easy readability.
            - For important terms like **Contract Sum**, **Employer**, etc., **bold** them for clarity.
            """
        elif filename == "prompts/chat_prompt.txt":
            return """
            Role: You are a helpful assistant specializing in UK construction contracts. Your audience is non-technical construction professionals.

            History:
            {history}

            Context:
            {context}
            
            FORMAT REQUIREMENTS:
            - Use CAPITALIZED HEADINGS for main sections (e.g., DOCUMENTS, PAYMENTS, RETENTION).
            - For each section, include an **Indented Subheading**: the prompt question from the left-hand column of the prompt table.
            - Provide **Bullet points** with answers extracted from the contract to the prompt question.
            - Use **natural language** and layman's terms.
            - Keep explanations brief and practical.
            - Use **bold** markdown for key dates, values, and timeframes (e.g., **35 days**, **March 2022**, **$10,000**).
            
            When information is not found in the contract:
            - State simply: "This contract doesn't specify..."
            - Suggest what the user might want to clarify.
            
            Question: {question}

            - Ensure that each section starts with the heading in **CAPITALIZED** format.
            - Ensure that each **subheading (question)** is clearly indented.
            - Each answer should be **bullet-pointed** for easy readability.
            - For important terms like **Contract Sum**, **Employer**, etc., **bold** them for clarity.
            """
        else:
            logger.error(f"Prompt file not found: {filename}")
            return ""




# Create prompt directory if it doesn't exist
Path("prompts").mkdir(exist_ok=True)

# Load prompts from files (or use defaults if files don't exist)
Summary_prompt = load_prompt_from_file("prompts/summary_prompt.txt")
chat_prompt = load_prompt_from_file("prompts/chat_prompt.txt")

# Create prompt templates
prompt_summary = ChatPromptTemplate.from_messages([
    ("system", Summary_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

prompt_chat = ChatPromptTemplate.from_messages([
    ("system", chat_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

output_parser = StrOutputParser()

def get_context_for_contract(contract_id, question):
    """Get relevant context for a contract based on a question"""
    retriever = get_retriever(contract_id)
    if not retriever:
        return "No documents found for this contract."
    
    try:
        docs = retriever.invoke(question)
        return format_docs(docs)
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving context from the contract."

# Chain for summary generation - optimized to retrieve context once
chain_summary = (
    {
        "context": RunnableLambda(
            lambda inputs: get_context_for_contract(inputs["contract_id"], inputs["question"])
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_summary
    | llm
    | output_parser
)

# Chain for chat interactions - optimized to retrieve context once
chain_chat = (
    {
        "context": RunnableLambda(
            lambda inputs: get_context_for_contract(inputs["contract_id"], inputs["question"])
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_chat
    | llm
    | output_parser
)

def get_message_history(contract_id: str) -> ChatMessageHistory:
    """Get or create a message history for a contract"""
    if contract_id not in message_histories:
        message_histories[contract_id] = ChatMessageHistory()
        message_histories[contract_id].add_ai_message("How can I help you?")
        # Add session creation time tracking
        session_creation_times[contract_id] = datetime.now()
    else:
        # Update session timestamp
        session_creation_times[contract_id] = datetime.now()
    return message_histories[contract_id]

chain_with_history_summary = RunnableWithMessageHistory(
    chain_summary,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history",
)

chain_with_history_chat = RunnableWithMessageHistory(
    chain_chat,
    get_message_history,
    input_messages_key="question",
    history_messages_key="history",
)

def process_pdf_chunks(contract_id, chunks):
    """Process PDF chunks and add to vector store"""
    vector_store_path = contract_vector_stores[contract_id]
    
    try:
        # Process in batches to avoid memory issues
        with get_chroma_db(contract_id) as vectordb:
            if not vectordb:
                logger.error(f"Failed to get vector DB for {contract_id}")
                processing_status[contract_id] = "failed"
                return False
                
            # Process in batches
            batch_size = BATCH_SIZE
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Add documents to Chroma
                vectordb.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1} of {len(chunks)//batch_size + 1} to vector store")
                
        return True
    except Exception as e:
        logger.error(f"Error processing PDF chunks: {e}")
        return False

def generate_full_summary(contract_id):
    """Generate a full summary for a contract"""
    try:
        query = "Provide the key contract details to generate the summary."
        
        # Use the chain with history to generate the summary
        response = chain_with_history_summary.invoke(
            {"question": "Generate a concise summary of this contract", "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def process_pdf_in_background(contract_id, file_path):
    """Background processing function for PDFs"""
    try:
        if contract_id not in contract_vector_stores:
            logger.error(f"Contract ID not found: {contract_id}")
            processing_status[contract_id] = "failed"
            return
            
        vector_store_path = contract_vector_stores[contract_id]
        processing_status[contract_id] = "processing"
        
        # Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            processing_status[contract_id] = "failed"
            logger.error(f"No content extracted from PDF for contract {contract_id}")
            return
            
        # Split the documents into smaller chunks
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to each document chunk
        for chunk in chunks:
            chunk.metadata["contract_id"] = contract_id
            
        # Process chunks
        success = process_pdf_chunks(contract_id, chunks)
        if not success:
            processing_status[contract_id] = "failed"
            return
        
        # Generate summary in background
        try:
            summary = generate_full_summary(contract_id)
            processing_status[contract_id] = "completed"
            
            # Store summary in message history
            if contract_id in message_histories:
                message_histories[contract_id] = ChatMessageHistory()
                message_histories[contract_id].add_ai_message(summary)
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            processing_status[contract_id] = "summary_failed"
            
        # Clean up temporary file
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                # Also clean up the parent temp directory if it's empty
                temp_dir = Path(file_path).parent
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {str(e)}")
            
    except Exception as e:
        processing_status[contract_id] = "failed"
        logger.error(f"Error processing PDF in background: {str(e)}")
        
        # Clean up on error
        try:
            if Path(vector_store_path).exists():
                shutil.rmtree(vector_store_path)
            if contract_id in contract_vector_stores:
                del contract_vector_stores[contract_id]
            if contract_id in chroma_cache:
                del chroma_cache[contract_id]
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up after failed processing: {str(cleanup_error)}")

# Define CORS preflight handlers for all routes
@app.route('/upload', methods=['OPTIONS'])
def upload_preflight():
    """Handle preflight CORS requests for upload endpoint"""
    response = jsonify({'message': 'Preflight request successful'})
    return response

@app.route('/chat', methods=['OPTIONS'])
def chat_preflight():
    """Handle preflight CORS requests for chat endpoint"""
    response = jsonify({'message': 'Preflight request successful'})
    return response

@app.route('/contracts', methods=['OPTIONS'])
def contracts_preflight():
    """Handle preflight CORS requests for contracts endpoint"""
    response = jsonify({'message': 'Preflight request successful'})
    return response

@app.route('/contract/<contract_id>', methods=['OPTIONS'])
def contract_preflight(contract_id):
    """Handle preflight CORS requests for contract endpoint"""
    response = jsonify({'message': 'Preflight request successful'})
    return response

@app.route('/contract/<contract_id>/summary', methods=['OPTIONS'])
def summary_preflight(contract_id):
    """Handle preflight CORS requests for summary endpoint"""
    response = jsonify({'message': 'Preflight request successful'})
    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload endpoint for PDF contracts with direct summary response"""
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        
        # Validate file
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files accepted"}), 400
            
        # Check file size
        file_size_mb = 0
        file.seek(0, os.SEEK_END)
        file_size_mb = file.tell() / (1024 * 1024)
        file.seek(0)  # Reset file pointer
        
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            return jsonify({
                "error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
            }), 400
        
        # Generate a new contract ID and create unique vector store directory
        contract_id = str(uuid.uuid4())
        vector_store_path = str(VECTOR_STORE_BASE_DIR / contract_id)
        Path(vector_store_path).mkdir(exist_ok=True)
        
        # Store the path mapping
        contract_vector_stores[contract_id] = vector_store_path
        
        # Track creation time for cleanup
        session_creation_times[contract_id] = datetime.now()
        
        # Create a temporary directory for processing
        temp_dir = TEMP_UPLOAD_DIR / contract_id
        temp_dir.mkdir(exist_ok=True)
        
        # Save the file with a sanitized filename
        safe_filename = Path(file.filename).name  # Get just the filename
        temp_path = str(temp_dir / safe_filename)
        file.save(temp_path)
        
        # Process in background
        processing_status[contract_id] = "started"
        
        # Process the document and get the summary synchronously
        # Instead of submitting to executor, we'll process directly
        try:
            # Load the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                processing_status[contract_id] = "failed"
                logger.error(f"No content extracted from PDF for contract {contract_id}")
                return jsonify({
                    "contract_id": contract_id,
                    "status": "failed",
                    "error": "Failed to extract content from PDF"
                }), 500
                
            # Split the documents into smaller chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata to each document chunk
            for chunk in chunks:
                chunk.metadata["contract_id"] = contract_id
                
            # Process chunks
            success = process_pdf_chunks(contract_id, chunks)
            if not success:
                processing_status[contract_id] = "failed"
                return jsonify({
                    "contract_id": contract_id,
                    "status": "failed",
                    "error": "Failed to process document chunks"
                }), 500
            
            # Generate summary
            summary = generate_full_summary(contract_id)
            processing_status[contract_id] = "completed"
            
            # Store summary in message history
            if contract_id in message_histories:
                message_histories[contract_id] = ChatMessageHistory()
                message_histories[contract_id].add_ai_message(summary)
                
            # Clean up temporary file
            try:
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                    # Also clean up the parent temp directory if it's empty
                    temp_dir = Path(temp_path).parent
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
                
            # Return the full summary with the response
            return jsonify({
                "contract_id": contract_id,
                "status": "completed",
                "summary": summary
            }), 200
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            processing_status[contract_id] = "failed"
            
            # Clean up on error
            try:
                if Path(vector_store_path).exists():
                    shutil.rmtree(vector_store_path)
                if contract_id in contract_vector_stores:
                    del contract_vector_stores[contract_id]
                if contract_id in chroma_cache:
                    del chroma_cache[contract_id]
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up after failed processing: {str(cleanup_error)}")
                
            return jsonify({
                "contract_id": contract_id,
                "status": "failed",
                "error": f"Error processing document: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        return jsonify({"error": f"Upload error: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat requests"""
    data = request.get_json()
    if not data or 'question' not in data or 'contract_id' not in data:
        return jsonify({"error": "Missing 'question' or 'contract_id' in the request body."}), 400

    contract_id = data['contract_id']
    question = data['question']
    
    # Validate input
    if len(question) > 5000:  # Limit question length
        return jsonify({"error": "Question too long"}), 400
    
    # Check if the contract exists
    if contract_id not in contract_vector_stores:
        return jsonify({"error": f"Contract ID not found: {contract_id}"}), 404
        
    # Check if processing is still in progress
    if contract_id in processing_status and processing_status[contract_id] != "completed":
        return jsonify({
            "processing": True,
            "message": "Document is still being processed. Please wait until processing is complete."
        }), 202
        
    try:
        # Check if vector store directory exists
        vector_store_path = contract_vector_stores[contract_id]
        if not Path(vector_store_path).exists():
            return jsonify({"error": f"Contract files not found for: {contract_id}"}), 404
            
        # Add the user's question to history
        get_message_history(contract_id).add_user_message(question)
        
        # Process the chat using the chain
        response = chain_with_history_chat.invoke(
            {"question": question, "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )
        
        # Add the response to history
        get_message_history(contract_id).add_ai_message(response)
        
        return jsonify({"answer": response}), 200
        
    except Exception as e:
        logger.error(f"Error occurred during chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/contracts', methods=['GET'])
def get_contracts():
    """Endpoint to list all uploaded contracts"""
    contracts_list = []
    
    # Make a copy of the keys to avoid modification during iteration
    for contract_id in contract_vector_stores.keys():
        try:
            # Get path and check existence
            vector_store_path = contract_vector_stores.get(contract_id)
            if not vector_store_path or not Path(vector_store_path).exists():
                # Skip this contract - cleanup will handle it separately
                continue
                
            # Get status
            status = processing_status.get(contract_id, "unknown")
            
            contract_info = {
                "contract_id": contract_id,
                "status": status
            }
            
            # Only include summaries for completed contracts
            if status == "completed" and contract_id in message_histories:
                history = message_histories[contract_id]
                # Get the most recent AI message as the summary (safely)
                ai_messages = [msg.content for msg in history.messages if msg.type == "ai"]
                if ai_messages:
                    # Truncate summary to a reasonable length
                    summary = ai_messages[-1][:500] + ("..." if len(ai_messages[-1]) > 500 else "")
                    contract_info["summary_preview"] = summary
            
            contracts_list.append(contract_info)
            
        except Exception as e:
            logger.error(f"Error processing contract {contract_id}: {str(e)}")
            continue

    if not contracts_list:
        return jsonify({"message": "No contracts found. Please upload documents to start."}), 404
        
    return jsonify(contracts_list), 200

@app.route('/contract/<contract_id>', methods=['GET'])
def check_contract(contract_id):
    """Check if a specific contract exists and its processing status"""
    try:
        if contract_id not in contract_vector_stores:
            return jsonify({"exists": False}), 404
            
        # Verify the directory actually exists
        vector_store_path = contract_vector_stores[contract_id]
        if not Path(vector_store_path).exists():
            return jsonify({"exists": False}), 404
            
        # Get status
        status = processing_status.get(contract_id, "unknown")
        
        # Prepare response
        response_data = {
            "exists": True,
            "status": status
        }
        
        # Include summary if available and requested
        include_summary = request.args.get('include_summary', 'false').lower() == 'true'
        if include_summary and status == "completed" and contract_id in message_histories:
            ai_messages = [msg.content for msg in message_histories[contract_id].messages if msg.type == "ai"]
            if ai_messages:
                response_data["summary"] = ai_messages[-1]
                
        return jsonify(response_data), 200
            
    except Exception as e:
        logger.error(f"Error checking contract {contract_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/contract/<contract_id>/summary', methods=['GET'])
def get_contract_summary(contract_id):
    """Get the full summary for a contract"""
    try:
        if contract_id not in contract_vector_stores:
            return jsonify({"error": "Contract not found"}), 404
            
        # Check if processing is complete
        status = processing_status.get(contract_id, "unknown")
        if status != "completed":
            return jsonify({
                "status": status,
                "message": "Contract processing is not complete"
            }), 202
            
        # Get the summary
        if contract_id in message_histories:
            ai_messages = [msg.content for msg in message_histories[contract_id].messages if msg.type == "ai"]
            if ai_messages:
                return jsonify({"summary": ai_messages[-1]}), 200
                
        # Generate a new summary if none exists
        summary = generate_full_summary(contract_id)
        return jsonify({"summary": summary}), 200
            
    except Exception as e:
        logger.error(f"Error getting summary for contract {contract_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def cleanup_old_sessions():
    """Remove sessions older than the configured timeout"""
    while True:
        try:
            current_time = datetime.now()
            contracts_to_remove = []
            
            # Identify old sessions
            for contract_id, creation_time in session_creation_times.items():
                if (current_time - creation_time) > timedelta(hours=SESSION_TIMEOUT_HOURS):
                    contracts_to_remove.append(contract_id)
                    
            # Process removals
            for contract_id in contracts_to_remove:
                logger.info(f"Cleaning up contract: {contract_id}")
                
                try:
                    # Remove vector store
                    if contract_id in contract_vector_stores:
                        vector_store_path = contract_vector_stores[contract_id]
                        if Path(vector_store_path).exists():
                            shutil.rmtree(vector_store_path)
                        del contract_vector_stores[contract_id]
                        
                    # Remove from cache
                    if contract_id in chroma_cache:
                        del chroma_cache[contract_id]
                        
                    # Remove session data
                    if contract_id in message_histories:
                        del message_histories[contract_id]
                    if contract_id in processing_status:
                        del processing_status[contract_id]
                except Exception as e:
                    logger.error(f"Error cleaning up contract {contract_id}: {str(e)}")
                    
            # Sleep for an hour before next cleanup
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            time.sleep(3600)  # Still sleep on error

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Log startup information
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Using vector store directory: {VECTOR_STORE_BASE_DIR}")
    logger.info(f"Using temporary upload directory: {TEMP_UPLOAD_DIR}")
    logger.info(f"Session timeout: {SESSION_TIMEOUT_HOURS} hours")

    # Use production WSGI server if available
    try:
        from waitress import serve
        logger.info("Using Waitress production server")
        serve(app, host=HOST, port=PORT)
    except ImportError:
        logger.warning("Waitress not installed, using Flask development server")
        app.run(host=HOST, port=PORT, threaded=True)