# RAG PDF Reader

This is a simple Retrieval-Augmented Generation (RAG) project that allows users to ask questions about the contents of a PDF file using OpenAI's GPT models. 
The system extracts text from the PDF, splits it into chunks, embeds the chunks, stores them in a FAISS vector index, and uses semantic retrieval to generate accurate, context-based answers.

## Features
- PDF text extraction using PyMuPDF
- Text chunking with overlap
- Vector storage and retrieval using FAISS
- Question answering using OpenAI's GPT model via LangChain

## Setup

1. Clone the repository:

git clone https://github.com/Meghana057/rag-pdf-reader-qa.git
cd rag-pdf-reader-qa

2.Install dependencies:

pip install -r requirements.txt

3.Add your OpenAI key in a .env file:

OPENAI_API_KEY=your_api_key_here
Add your PDF (rename to sample.pdf or update in pdf-qa.py)

4.Run:
python pdf-qa.py

## Example questions
Example Questions: 

1. Who won the race?
2. What is the moral of the story?
3. Why did the hare fall asleep?
