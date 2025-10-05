import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


class RAGPipeline:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')
        self.llm = OllamaLLM(model='deepseek-v3.1:671b-cloud')
        self.qa_chain = None

    def load_file(self, uploaded_file):
        """Load PDF, DOCX, or TXT."""
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        docs = loader.load()
        os.remove(file_path)
        return docs

    def build_retriever(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vector_storage = FAISS.from_documents(chunks, self.embeddings)
        return vector_storage.as_retriever()

    def create_qa_chain(self, retriever):
        system_msg = SystemMessagePromptTemplate.from_template(
            "You are GramVani — a helpful and concise AI assistant. "
            "Use the provided document content to answer the question. "
            "If the document doesn't contain the answer, rely on your general knowledge. "
            "Always respond clearly and under 500 words."
        )

        human_msg = HumanMessagePromptTemplate.from_template(
            "Document Content:\n{context}\n\nQuestion:\n{input}"
        )

        chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        document_chain = create_stuff_documents_chain(self.llm, chat_prompt)
        self.qa_chain = create_retrieval_chain(retriever, document_chain)

    def ask(self, query, output_language="English"):
        if self.qa_chain:
            response = self.qa_chain.invoke({"input": query})
            answer = response["answer"]
        else:
            general_prompt = f"""
                You are GramVani — a helpful and concise AI assistant.
                Answer the question in {output_language} clearly and concisely.
                Question: {query}
            """
            answer = self.llm.invoke(general_prompt)

        if output_language.lower() != "english":
            translate_prompt = f"Translate this answer into {output_language} concisely:\n{answer}"
            answer = self.llm.invoke(translate_prompt)

        return answer
