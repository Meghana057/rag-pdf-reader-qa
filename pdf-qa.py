import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Get text from PDF 
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Split 
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

# Embed and index
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

#  QA 
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", k=5)
    llm = ChatOpenAI(temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


def main():
    pdf_path = "sample.pdf"
    print("[+] Reading PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("[+] Splitting text...")
    chunks = split_text(text)

    print("[+] Creating vector store...")
    vector_store = create_vector_store(chunks)

    print("[+] Setting up QA chain...")
    qa_chain = create_qa_chain(vector_store)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})  
        print("\n[Answer]")
        print(result["result"])

        print("\n[Sources]")
        for doc in result["source_documents"]:
            print("-", doc.page_content[:200], "...")  


if __name__ == "__main__":
    main()
