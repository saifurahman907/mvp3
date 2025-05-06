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


if os.path.exists("vector_store"):
    shutil.rmtree("vector_store")


load_dotenv()
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
CORS(app)

persist_directory = "vector_store"
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500, chunk_overlap=300)

# Initialize Chroma and ensure that we can store vector data
try:
    # Initialize Chroma vector store with persist_directory
    if os.path.exists(persist_directory):
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
    else:
        os.makedirs(persist_directory, exist_ok=True)
        vectordb = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
    print(f"Number of documents in the vector store: {len(vectordb.get()['documents'])}")


except Exception as e:
    print("Error initializing Chroma:", str(e))
    raise


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def print_me(x):
    print(x)
    return x


def get_retriever(contract_id):
    num_docs = len(vectordb.get()["documents"])
    if num_docs == 0:
        return None  # or handle appropriately
    return vectordb.as_retriever(search_kwargs={'filter': {'contract_id': contract_id}, "k": 35})


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
            )
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
            )
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


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        # Validate file
        if not file or file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files accepted"}), 400
        
        # We'll use the already initialized vectordb instance
        print("Using existing Chroma instance")

        # Ensure the uploaded PDF file is processed into documents and split into chunks
        contract_id = str(uuid.uuid4())
        temp_dir = os.path.join("temp_uploads", contract_id)
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)

        file.save(temp_path)  # Save the uploaded file

        # Validate file size
        if os.path.getsize(temp_path) == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # Load the PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")

        # Check if documents were loaded successfully
        if not documents:
            return jsonify({"error": "PDF text extraction failed - document may be scanned or encrypted"}), 400

        # Split the documents into smaller chunks
        texts = text_splitter.split_documents(documents)
        print(f"Created {len(texts)} chunks from the documents.")


        # Ensure that the metadata is set for each document
        for doc in texts:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["contract_id"] = contract_id

        # Add documents to the Chroma vector store
        vectordb.add_documents(texts)
        print("Documents added to Chroma successfully")

        # Check how many documents are now in the vector store
        print(f"Number of documents in the vector store: {len(vectordb.get()['documents'])}")

        # Generate a summary for the document
        summary = chain_with_history_summary.invoke(
            {"question": "Generate a full contract breakdown covering all sections...", "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )
        print(f"Generated summary: {summary}")


        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)  # Clean up temp files
        except Exception as e:
            print(f"Error cleaning up temp directory: {e}")

        # Return contract_id and the generated summary in the response
        return jsonify({
            "contract_id": contract_id,
            "summary": summary  # Now returns the summary instead of the default message
        }), 200
    
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Add more detailed error logging
        return jsonify({"error": str(e)}), 500



# Handle chat requests
@app.route('/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    if not data or 'question' not in data or 'contract_id' not in data:
        return jsonify({"error": "Missing 'question' or 'contract_id' in the request body."}), 400

    contract_id = data['contract_id']

    # Check if the vector store is empty
    try:
        documents = vectordb.get()["documents"]
        if len(documents) == 0:
            return jsonify({"error": f"No documents found for contract ID: {contract_id}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error accessing vector store: {str(e)}"}), 500


    try:
        response = chain_with_history_chat.invoke(
            {"question": data['question'], "contract_id": contract_id},
            config={"configurable": {"session_id": contract_id}}
        )
        return jsonify({"answer": response}), 200
    except Exception as e:
        print(f"Error occurred: {e}")  # Log the full error to the console for debugging
        return jsonify({"error": str(e)}), 500


# List all contracts
@app.route('/contracts', methods=['GET'])
def get_contracts():
    """Endpoint to list all uploaded contracts"""
    if not message_histories:
        return jsonify({"message": "No contracts found. Please upload documents to start."}), 404

    contracts_list = [
        {
            "contract_id": contract_id,
            "summary": next((msg.content for msg in history.messages if msg.type == "ai"), "No summary available")
        }
        for contract_id, history in message_histories.items()
    ]

    return jsonify(contracts_list), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)