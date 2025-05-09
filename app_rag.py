from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
# Updated imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# Corrected import for RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use the correct import for output parsing
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import uuid
from dotenv import load_dotenv
from datetime import datetime, timedelta
import threading
import time
import shutil
from langchain_core.runnables import RunnableLambda


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
# Improved CORS configuration
CORS(app)

# Base directory for vector stores
VECTOR_STORE_BASE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_BASE_DIR, exist_ok=True)

# Dictionary to store contract_id -> vector_store_path mapping
contract_vector_stores = {}

# Initialize embeddings and text splitter
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=300)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def print_me(x):
    print(x)
    return x


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
            
        return vectordb.as_retriever(search_kwargs={"k": 35})
    except Exception as e:
        print(f"Error getting retriever for contract {contract_id}: {str(e)}")
        # If there's an error, clean up the tracking
        if contract_id in contract_vector_stores:
            del contract_vector_stores[contract_id]
        return None


# LLM and prompt
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

Summary_prompt = """
                History:
                {history}

                Context:
                {context}

                Role: You are a contract law expert specializing in UK construction contracts. Your audience is made up of non-technical construction professionals who are not experts in contracts.

                Task: Using the provided context, produce a detailed summary of the contract. Write in clear, simple, everyday language, and explain any technical terms so that a layperson can easily understand. Your summary should include all the essential points and explanations in natural language with reasonable context.

                FORMAT REQUIREMENTS:
                - Use CAPITALIZED HEADINGS for main sections exactly as listed in the template (e.g., DOCUMENTS, CONTRACT FORM)
                - Under each heading, include the specific question as an indented sub-heading exactly as written in the template
                - Present answers as bullet points with reasonable context
                - Keep explanations brief and in layman's terms
                - Bold key dates, values, and timeframes using **bold** markdown

                Please address the following areas:

                1. DOCUMENTS
                    What documents are included in this contract?
                    • List all the uploaded documents (e.g. contract, order, minutes, etc.).

                2. CONTRACT FORM
                    What form of contract is being used?
                    • Identify the form of contract (e.g. Contract Noggin 2016 Design & Build, Contract Noggin Intermediate 2016, etc.).

                3. PAYMENTS
                    What are the payment terms?
                    • Describe the payment terms in simple language.
                    
                    How long before the final due date can a pay-less notice be issued?
                    • Explain with exact number of days and cite relevant sections.

                4. TERMINATION
                    What are the termination provisions?
                    • Summarize the termination clauses.
                    
                    What costs are involved if the contract is terminated?
                    • Explain the financial implications of termination.

                5. SUSPENSION
                    Under what circumstances can the works be suspended?
                    • Clarify the suspension conditions.
                    
                    What notice is required for suspension?
                    • Detail notice periods and required information.
                    
                    Can the subcontractor charge for resuming work after suspension?
                    • Explain costs and limitations.

                6. VARIATIONS
                    How are contract variations handled?
                    • Summarize variation clauses.
                    
                    What is the time limit for submitting a variation?
                    • Specify exact timeframes.
                    
                    Is prior approval needed before starting variations?
                    • Clarify approval requirements.
                    
                    Who can authorize variations?
                    • Identify authorized parties.
                    
                    Must the subcontractor proceed without prior sign-off?
                    • State conditions for proceeding and payment risks.

                7. DAY WORKS
                    What are the daywork rates and calculations?
                    • Describe rates and percentage calculations.
                    
                    What is included in these rates?
                    • Explain inclusions (supervisors, labor, plant) and exclusions.

                8. EXTENSIONS OF TIME
                    What are valid grounds for extension of time?
                    • Outline qualifying circumstances.
                    
                    What information must be included in an Extension of Time submission?
                    • Summarize required documentation.

                9. RETENTION
                    What percentage of retention is held?
                    • State the retention percentage.
                    
                    What is the total retention amount?
                    • Calculate if contract sum is provided.
                    
                    How long is the defects period?
                    • Explain duration and start timing.

                10. ADJUDICATION
                    Does the subcontractor have adjudication rights?
                    • Clarify adjudication provisions.
                    
                    Are adjudicator fees fixed or capped?
                    • Provide fee structure if specified.

                11. ENTIRE AGREEMENT CLAUSE
                    What documents form the entire agreement?
                    • Explain which documents are contractually binding.

                12. PROGRAMME
                    What is the weekly value of work required?
                    • Calculate based on programme duration and contract sum.
                    
                    Is the programme a numbered document?
                    • Confirm document status.
                    
                    What is the value of Liquidated and Ascertained Damages?
                    • Identify LAD values.
                    
                    Do LADs exceed 1% for 10 weeks maximum?
                    • Assess against recommended thresholds.

                13. KEY RISKS
                    What are the main risks in this contract?
                    • Highlight significant risks in plain language.

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
                - Use CAPITALIZED HEADINGS for main sections exactly as listed in the template
                - Under each heading, include the specific question as an indented sub-heading exactly as written
                - Present answers as bullet points with reasonable context
                - Use natural language and layman's terms
                - Keep explanations brief and practical
                - Bold key dates, values, and timeframes using **bold** markdown
                
                When information is not found in the contract:
                - State simply: "This contract doesn't specify..."
                - Suggest what the user might want to clarify
                - Provide a brief explanation of what would typically be expected
                
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

# Use StrOutputParser instead of RetryOutputParser
output_parser = StrOutputParser()

# Use the correct output parser with the chains
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
    | output_parser  # Use StrOutputParser
)

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
    | output_parser  # Use StrOutputParser here as well
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

# Improved upload handler with better error tracking
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
        temp_dir = os.path.join("temp_uploads", contract_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        # Load the PDF
        try:
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                # Clean up on error
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if os.path.exists(vector_store_path):
                    shutil.rmtree(vector_store_path)
                if contract_id in contract_vector_stores:
                    del contract_vector_stores[contract_id]
                if contract_id in session_creation_times:
                    del session_creation_times[contract_id]
                return jsonify({"error": "PDF text extraction failed"}), 400
                
            # Split the documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=300)
            texts = text_splitter.split_documents(documents)
            
            # Add metadata to each document chunk
            for text in texts:
                text.metadata["contract_id"] = contract_id
                
            # Initialize a new vector store for this contract
            vectordb = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embedding
            )
            
            # Add documents to Chroma
            vectordb.add_documents(texts)
            print(f"Added {len(texts)} documents to vector store at {vector_store_path}")
            
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
            # Generate full summary
            summary = generate_full_summary(contract_id)
            
            return jsonify({"contract_id": contract_id, "summary": summary}), 200
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(vector_store_path):
                shutil.rmtree(vector_store_path)
            if contract_id in contract_vector_stores:
                del contract_vector_stores[contract_id]
            if contract_id in session_creation_times:
                del session_creation_times[contract_id]
            return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

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
            retriever = vectordb.as_retriever(search_kwargs={"k": 35})
            
            # Use the newer invoke() method instead of get_relevant_documents()
            query = "Provide the full contract details to generate the summary."
            
            try:
                # Try the newer method first
                documents = retriever.invoke(query)
            except Exception as e:
                print(f"Error using invoke method: {str(e)}")
                # Fall back to the deprecated method
                documents = retriever.get_relevant_documents(query)
            
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
                    question="Generate a comprehensive summary of this contract",
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
            continue
            
        # Get history if it exists
        if contract_id in message_histories:
            history = message_histories[contract_id]
            # Get the most recent AI message as the summary
            ai_messages = [msg.content for msg in history.messages if msg.type == "ai"]
            summary = ai_messages[-1] if ai_messages else "No summary available"
            
            contracts_list.append({
                "contract_id": contract_id,
                "summary": summary
            })

    if not contracts_list:
        return jsonify({"message": "No contracts found. Please upload documents to start."}), 404
        
    return jsonify(contracts_list), 200


# Endpoint to check contract existence
@app.route('/contract/<contract_id>', methods=['GET'])
def check_contract(contract_id):
    """Check if a specific contract exists"""
    if contract_id in contract_vector_stores:
        # Verify the directory actually exists
        vector_store_path = contract_vector_stores[contract_id]
        if os.path.exists(vector_store_path):
            return jsonify({"exists": True}), 200
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
    app.run(host='0.0.0.0', port=5000)