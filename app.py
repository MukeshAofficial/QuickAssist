import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from together import Together

# Define the Together API key
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
client = Together(api_key=TOGETHER_API_KEY)

# Function to load and process documents from a list of uploaded files
def load_documents_from_files(files):
    documents = []
    for file in files:
        # Save the uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())  # Write the content of the uploaded file
            temp_file_path = temp_file.name  # Get the path of the saved file
        # Use the saved file path with PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        documents.extend(docs)
        # Clean up the temporary file after processing
        os.remove(temp_file_path)
    return documents

# Function to process the documents and create the FAISS vector store
def create_vector_store_from_files(files):
    # Load and process documents from the uploaded files
    documents = load_documents_from_files(files)
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Use Sentence Transformers for embeddings
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store using the SentenceTransformer embeddings
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    
    return vectorstore

# Create a retriever function to get relevant documents based on the query
def retrieve_information(query, files):
    # Create the vector store only once when the files are uploaded
    vectorstore = create_vector_store_from_files(files)
    
    # Retrieve information based on the query
    retriever = vectorstore.as_retriever()
    retrieved_info = retriever.invoke(query)
    
    # Return the first relevant document's content or a message if no results
    if retrieved_info:
        return retrieved_info[0].page_content
    else:
        return "No relevant information found."

# Function to send the retrieved context and query to the LLM and get the answer
def query_llm(query, retrieved_text):
    prompt = f"""
    The user is asking the following question: "{query}"
    Below is some relevant information from the documents that could help in answering the question:
    {retrieved_text}

    Now, based on the information above, please provide a short and human-like answer to the user query. Limit your response to a few lines, summarizing the key points mostly from the given context.
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,  # Control the response length
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True
        )

        full_response = ""
        for token in response:
            if hasattr(token, 'choices'):
                full_response += token.choices[0].delta.content
        
        return full_response

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit App
st.title("Customer Support AI")
st.markdown("Upload your company's knowledge base and get instant 24/7 customer support")

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Display User's query
    query = st.text_input("Ask a question:")

    if query:
        # Show the user's question
        st.markdown(f"**User:** {query}")

        # Call the function to retrieve information based on the query
        retrieved_info = retrieve_information(query, uploaded_files)
        
        # Call the LLM with the retrieved information and the user's query
        answer = query_llm(query, retrieved_info)

        # Display Chatbot's response
        st.markdown(f"**Chatbot:** {answer}")
