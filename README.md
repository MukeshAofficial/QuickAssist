# QuickAssist
AI-powered customer support agent that provides instant, 24/7 assistance using your company's knowledge base. It quickly resolves queries, streamlines support, and improves customer satisfaction by efficiently handling PDF-based queries and offering intelligent, tailored responses.

This project is a PDF Query Chatbot that allows users to upload PDF files, process the content, and ask questions. It leverages open-source models such as Meta-Llama-3.1-405B-Instruct-Turbo for answering user queries and all-MiniLM-L6-v2 for generating embeddings. The chatbot is deployed using Streamlit, providing an interactive web interface.

## Features

- **Upload PDF Files**: Users can upload multiple PDF files.
- **Ask Questions**: Users can ask questions related to the uploaded documents.
- **Instant Answers**: The chatbot retrieves relevant information from the documents and provides human-like responses.
- **Open-Source Models**: Utilizes the Meta-Llama-3.1-405B-Instruct-Turbo for query answering and all-MiniLM-L6-v2 for document embeddings.
- **Deployable on Streamlit**: The app is deployable on Streamlit for a user-friendly web interface.

### 1. Set your `TOGETHER_API_KEY` environment variable

In your code, set the API key as follows:

```python
import os
TOGETHER_API_KEY = "your-api-key"
os.getenv(TOGETHER_API_KEY)
```

Run the Streamlit app
Once the environment variable is set, you can run the Streamlit app with the following command:
```python
streamlit run app.py
```

