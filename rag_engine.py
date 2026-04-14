import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.vector_store = None
        # Stable local embeddings to avoid cloud API model availability errors
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Fast free cloud LLM via Groq API
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    def ingest_document(self, file_path: str):
        print(f"Loading document from {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        print(f"Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if not splits:
            raise ValueError("No readable text could be extracted from this PDF. If it's a scanned image, it requires OCR.")

        print(f"Creating vector store with {len(splits)} chunks...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        print("Vector store created successfully.")

    def ask_question(self, question: str) -> str:
        if not self.vector_store:
            return "Please upload a document first before asking a question."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        
        system_prompt = (
            "You are a helpful and intelligent assistant for document analysis. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "You are encouraged to synthesize summaries, extract topics, and infer answers based on the context. "
            "If the question asks for general information like titles or topics, use the context to deduce the best possible answer. "
            "If you truly cannot deduce the answer from the context at all, say: 'I am sorry, but the answer to this question was not found in the uploaded document.' "
            "Format your answers clearly using Markdown bullet points or numbered lists where appropriate to make them easy to read. "
            "Be detailed, professional, and provide as much relevant information as possible from the text.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        results = rag_chain.invoke({"input": question})
        return results["answer"]
