from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import shutil
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# For asynchronous processing
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if not api_key:
    print("API key not found. Please check your .env file.")
else:
    print("API key loaded successfully!")

app = Flask(__name__)
# Configure CORS with appropriate settings
CORS(app, resources={r"/*": {"origins": "*"}})

# Base directory for vector stores
VECTOR_STORE_BASE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_BASE_DIR, exist_ok=True)

# Temporary upload directory
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Dictionary to store contract_id -> vector_store_path mapping
contract_vector_stores = {}

# Dictionary to track processing status
processing_status = {}

# Initialize embeddings and text splitter
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Reduced chunk size for faster processing
    chunk_overlap=100  # Reduced overlap
)

# Create a thread pool for background processing
executor = ThreadPoolExecutor(max_workers=5)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Improved Retriever Function
def get_retriever(contract_id):
    # Get the vector store path for this contract
    if contract_id not in contract_vector_stores:
        print(f"Error: No vector store found for contract ID: {contract_id}")
        return None
        
    vector_store_path = contract_vector_stores[contract_id]
    
    # Check if the directory exists
    if not os.path.exists(vector_store_path):
        print(f"Error: Vector store directory does not exist: {vector_store_path}")
        # Remove from tracking if directory doesn't exist
        if contract_id in contract_vector_stores:
            del contract_vector_stores[contract_id]
        return None
    
    # Initialize the vector store for this contract
    try:
        vectordb = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedding
        )
        
        # Check if there are documents in the vector store
        docs_info = vectordb.get()
        if len(docs_info["documents"]) == 0:
            print(f"Warning: No documents in vector store for contract ID: {contract_id}")
            return None
            
        return vectordb.as_retriever(search_kwargs={"k": 20})  # Reduced k for faster retrieval
    except Exception as e:
        print(f"Error getting retriever for contract {contract_id}: {str(e)}")
        # If there's an error, clean up the tracking
        if contract_id in contract_vector_stores:
            del contract_vector_stores[contract_id]
        return None

# LLM with timeout configuration
llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=api_key,
    request_timeout=90,  # Increased timeout for API calls
    max_retries=3  # Add retries for resilience
)

# Summary prompt (shortened for brevity)
Summary_prompt = """
                History:
                {history}

                Context:
                {context}

                Role: You are a contract law expert specializing in UK construction contracts. Your audience is made up of non-technical construction professionals who are not experts in contracts.

                Task: Using the provided context, produce a detailed summary of the contract. Write in clear, simple, everyday language, and explain any technical terms so that a layperson can easily understand.

                FORMAT REQUIREMENTS:
                - Use CAPITALIZED HEADINGS for main sections
                - Present answers as bullet points with reasonable context
                - Keep explanations brief and in layman's terms
                - Bold key dates, values, and timeframes using **bold** markdown

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
                """

chat_prompt = """
                Role: You are a helpful assistant specializing in UK construction contracts. Your audience is non-technical construction professionals.

                History:
                {history}

                Context:
                {context}
                
                FORMAT REQUIREMENTS:
                - Use CAPITALIZED HEADINGS for main sections
                - Present answers as bullet points with reasonable context
                - Use natural language and layman's terms
                - Keep explanations brief and practical
                
                When information is not found in the contract:
                - State simply: "This contract doesn't specify..."
                - Suggest what the user might want to clarify
                
                Question: {question}
                """

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

# Chain for summary generation
chain_summary = (
    {
        "context": RunnableLambda(
            lambda inputs: format_docs(
                get_retriever(inputs["contract_id"]).invoke(inputs["question"])
            ) if get_retriever(inputs["contract_id"]) else "No documents found for this contract."
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_summary
    | llm
    | output_parser
)

# Chain for chat interactions
chain_chat = (
    {
        "context": RunnableLambda(
            lambda inputs: format_docs(
                get_retriever(inputs["contract_id"]).invoke(inputs["question"])
            ) if get_retriever(inputs["contract_id"]) else "No documents found for this contract."
        ),
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt_chat
    | llm
    | output_parser
)

# Session management
message_histories = {}
session_creation_times = {}

def get_message_history(contract_id: str) -> ChatMessageHistory:
    if contract_id not in message_histories:
        message_histories[contract_id] = ChatMessageHistory()
        message_histories[contract_id].add_ai_message("How can I help you?")
        # Add session creation time tracking
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

# Background processing function for PDFs
def process_pdf_in_background(contract_id, file_path):
    try:
        vector_store_path = contract_vector_stores[contract_id]
        processing_status[contract_id] = "processing"
        
        # Load the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            processing_status[contract_id] = "failed"
            print(f"No content extracted from PDF for contract {contract_id}")
            return
            
        # Split the documents into smaller chunks
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to each document chunk
        for chunk in chunks:
            chunk.metadata["contract_id"] = contract_id
            
        # Process in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Initialize vector store
            vectordb = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding
            )
            
            # Add documents to Chroma
            vectordb.add_documents(batch)
            print(f"Added batch {i//batch_size + 1} of {len(chunks)//batch_size + 1} to vector store")
        
        # Generate summary in background
        try:
            summary = generate_full_summary(contract_id)
            processing_status[contract_id] = "completed"
            
            # Store summary in message history
            if contract_id in message_histories:
                message_histories[contract_id] = ChatMessageHistory()
                message_histories[contract_id].add_ai_message(summary)
                
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            processing_status[contract_id] = "summary_failed"
            
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        processing_status[contract_id] = "failed"
        print(f"Error processing PDF in background: {str(e)}")
        
        # Clean up on error
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)
        if contract_id in contract_vector_stores:
            del contract_vector_stores[contract_id]

# Upload endpoint with streaming processing
# Upload endpoint with summary in response
@app.route('/upload', methods=['POST'])
def upload_file():
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
        
        # Generate a new contract ID and create unique vector store directory
        contract_id = str(uuid.uuid4())
        vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, contract_id)
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Store the path mapping
        contract_vector_stores[contract_id] = vector_store_path
        
        # Track creation time for cleanup
        session_creation_times[contract_id] = datetime.now()
        
        # Create a temporary directory for processing
        temp_dir = os.path.join(TEMP_UPLOAD_DIR, contract_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        # Process PDF synchronously to get the summary immediately
        try:
            # Load the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                return jsonify({"error": "No content extracted from PDF"}), 400
                
            # Split the documents into smaller chunks
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata to each document chunk
            for chunk in chunks:
                chunk.metadata["contract_id"] = contract_id
                
            # Process in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Initialize vector store
                vectordb = Chroma(
                    persist_directory=vector_store_path,
                    embedding_function=embedding
                )
                
                # Add documents to Chroma
                vectordb.add_documents(batch)
            
            # Generate summary immediately
            summary = generate_full_summary(contract_id)
            processing_status[contract_id] = "completed"
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Return with contract_id and full summary
            return jsonify({
                "contract_id": contract_id,
                "summary": summary
            }), 200
                
        except Exception as e:
            # If immediate processing fails, fall back to background processing
            processing_status[contract_id] = "started"
            executor.submit(process_pdf_in_background, contract_id, temp_path)
            
            return jsonify({
                "contract_id": contract_id, 
                "message": "Document processing in progress. Summary not immediately available.",
                "status": "processing"
            }), 202
            
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# Status is now checked through the /contract endpoint

# Fixed Summary Generation Function
def generate_full_summary(contract_id):
    try:
        # Check if the contract exists
        if contract_id not in contract_vector_stores:
            return "Contract not found."
            
        vector_store_path = contract_vector_stores[contract_id]
        
        # Check if the directory exists
        if not os.path.exists(vector_store_path):
            print(f"Error: Vector store directory does not exist: {vector_store_path}")
            if contract_id in contract_vector_stores:
                del contract_vector_stores[contract_id]
            return "Contract files not found."
        
        # Initialize vector store for this contract
        try:
            vectordb = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding
            )
            
            # Create a retriever for this contract
            retriever = vectordb.as_retriever(search_kwargs={"k": 20})  # Reduced k for faster retrieval
            
            # Use the newer invoke() method
            query = "Provide the key contract details to generate the summary."
            documents = retriever.invoke(query)
            
            if not documents:
                return "No documents found for the given contract."
                
            # Format the documents to create the context
            context = format_docs(documents)
            
            # Get or create message history for this contract
            if contract_id not in message_histories:
                message_histories[contract_id] = ChatMessageHistory()
                # Track creation time
                session_creation_times[contract_id] = datetime.now()
                
            # Create message placeholder for the history
            history_messages = message_histories[contract_id].messages
            
            # Use the LLM to generate the summary
            response = llm.invoke(
                prompt_summary.format(
                    question="Generate a concise summary of this contract",
                    context=context,
                    history=history_messages
                )
            )
            
            # Save the summary as the first AI message in the history
            if contract_id in message_histories:
                # Clear existing history and add new summary
                message_histories[contract_id] = ChatMessageHistory()
                message_histories[contract_id].add_ai_message(response.content)
                # Reset session creation time
                session_creation_times[contract_id] = datetime.now()
            
            return response.content
            
        except Exception as e:
            print(f"Error with vector store: {str(e)}")
            return f"Error accessing contract data: {str(e)}"
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Handle chat requests
@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    if not data or 'question' not in data or 'contract_id' not in data:
        return jsonify({"error": "Missing 'question' or 'contract_id' in the request body."}), 400

    contract_id = data['contract_id']
    
    # Check if the contract exists
    if contract_id not in contract_vector_stores:
        return jsonify({"error": f"Contract ID not found: {contract_id}"}), 404
        
    # Check if processing is still in progress
    if contract_id in processing_status and processing_status[contract_id] != "completed":
        return jsonify({
            "processing": True,
            "message": "Document is still being processed. Please wait until processing is complete."
        }), 202  # 202 Accepted indicates the request was valid but processing is not complete
        
    try:
        # Check if vector store directory exists
        vector_store_path = contract_vector_stores[contract_id]
        if not os.path.exists(vector_store_path):
            return jsonify({"error": f"Contract files not found for: {contract_id}"}), 404
            
        # Add the user message to history
        if contract_id not in message_histories:
            message_histories[contract_id] = ChatMessageHistory()
            message_histories[contract_id].add_ai_message("How can I help you?")
            # Track creation time
            session_creation_times[contract_id] = datetime.now()
        else:
            # Update the session timestamp
            session_creation_times[contract_id] = datetime.now()
        
        # Add the user's question to history
        message_histories[contract_id].add_user_message(data['question'])
        
        # Process the chat using the chain
        response = chain_with_history_chat.invoke(
            {"question": data['question'], "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )
        
        # Add the response to history
        message_histories[contract_id].add_ai_message(response)
        
        return jsonify({"answer": response}), 200
        
    except Exception as e:
        print(f"Error occurred during chat: {e}")
        return jsonify({"error": str(e)}), 500

# List all contracts
@app.route('/contracts', methods=['GET'])
def get_contracts():
    """Endpoint to list all uploaded contracts"""
    contracts_list = []
    
    for contract_id in list(contract_vector_stores.keys()):
        # Check if vector store path exists
        vector_store_path = contract_vector_stores[contract_id]
        if not os.path.exists(vector_store_path):
            # Remove from tracking if directory doesn't exist
            del contract_vector_stores[contract_id]
            if contract_id in message_histories:
                del message_histories[contract_id]
            if contract_id in session_creation_times:
                del session_creation_times[contract_id]
            if contract_id in processing_status:
                del processing_status[contract_id]
            continue
            
        # Get status
        status = processing_status.get(contract_id, "unknown")
        
        # Only include fully processed contracts with summaries
        if status == "completed" and contract_id in message_histories:
            history = message_histories[contract_id]
            # Get the most recent AI message as the summary
            ai_messages = [msg.content for msg in history.messages if msg.type == "ai"]
            summary = ai_messages[-1] if ai_messages else "No summary available"
            
            contracts_list.append({
                "contract_id": contract_id,
                "summary": summary
            })
        else:
            # Include contracts that are still processing, but without summary
            contracts_list.append({
                "contract_id": contract_id,
                "status": status
            })

    if not contracts_list:
        return jsonify({"message": "No contracts found. Please upload documents to start."}), 404
        
    return jsonify(contracts_list), 200

# Endpoint to check contract existence and processing status
@app.route('/contract/<contract_id>', methods=['GET'])
def check_contract(contract_id):
    """Check if a specific contract exists and its processing status"""
    if contract_id in contract_vector_stores:
        # Verify the directory actually exists
        vector_store_path = contract_vector_stores[contract_id]
        if os.path.exists(vector_store_path):
            # Get status
            status = processing_status.get(contract_id, "unknown")
            
            # Get summary if processing is complete
            summary = None
            if status == "completed" and contract_id in message_histories:
                ai_messages = [msg.content for msg in message_histories[contract_id].messages if msg.type == "ai"]
                summary = ai_messages[-1] if ai_messages else None
                
            response_data = {
                "exists": True,
                "status": status
            }
            
            # Include summary if available
            if summary:
                response_data["summary"] = summary
                
            return jsonify(response_data), 200
        else:
            # Clean up tracking for non-existent directory
            del contract_vector_stores[contract_id]
            if contract_id in message_histories:
                del message_histories[contract_id]
            if contract_id in session_creation_times:
                del session_creation_times[contract_id]
            return jsonify({"exists": False}), 404
    else:
        return jsonify({"exists": False}), 404

# Cleanup function to remove old sessions
def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    while True:
        try:
            current_time = datetime.now()
            contracts_to_remove = []
            
            for contract_id, creation_time in list(session_creation_times.items()):
                if (current_time - creation_time) > timedelta(hours=24):
                    contracts_to_remove.append(contract_id)
                    
            for contract_id in contracts_to_remove:
                print(f"Cleaning up contract: {contract_id}")
                
                # Remove session data
                if contract_id in message_histories:
                    del message_histories[contract_id]
                if contract_id in session_creation_times:
                    del session_creation_times[contract_id]
                if contract_id in processing_status:
                    del processing_status[contract_id]
                    
                # Remove vector store
                if contract_id in contract_vector_stores:
                    vector_store_path = contract_vector_stores[contract_id]
                    if os.path.exists(vector_store_path):
                        shutil.rmtree(vector_store_path)
                    del contract_vector_stores[contract_id]
                    
            # Sleep for an hour before next cleanup
            time.sleep(3600)
            
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            time.sleep(3600)  # Still sleep on error

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Use threaded=True for better handling of concurrent requests
    app.run(host='0.0.0.0', port=5000, threaded=True)