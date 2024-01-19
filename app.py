import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import pandas as pd
from docx import Document

def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_docx(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    return text

def get_text_from_csv(csv_docs):
    text = ""
    for csv_file in csv_docs:
        df = pd.read_csv(csv_file)
        text += df.to_string(index=False) + "\n"
    return text

def get_text_from_xlsx(xlsx_docs):
    text = ""
    for xlsx_file in xlsx_docs:
        df = pd.read_excel(xlsx_file)
        text += df.to_string(index=False) + "\n"
    return text

def get_text_from_txt(txt_docs):
    text = ""
    for txt_file in txt_docs:
        text += txt_file.getvalue().decode("utf-8") + "\n"
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Documents",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
        st.subheader("Your documents")
        all_docs = st.file_uploader(
            "Upload your files here", accept_multiple_files=True,
            type=['pdf', 'docx', 'csv', 'xlsx', 'txt']
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""

                if all_docs:
                    pdf_docs = [doc for doc in all_docs if doc.type == 'application/pdf']
                    docx_docs = [doc for doc in all_docs if doc.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
                    csv_docs = [doc for doc in all_docs if doc.type == 'text/csv']
                    xlsx_docs = [doc for doc in all_docs if doc.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
                    txt_docs = [doc for doc in all_docs if doc.type == 'text/plain']

                    if pdf_docs:
                        raw_text += get_text_from_pdf(pdf_docs)

                    if docx_docs:
                        raw_text += get_text_from_docx(docx_docs)

                    if csv_docs:
                        raw_text += get_text_from_csv(csv_docs)

                    if xlsx_docs:
                        raw_text += get_text_from_xlsx(xlsx_docs)

                    if txt_docs:
                        raw_text += get_text_from_txt(txt_docs)    

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    
                    st.success("Done!!")    

    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()
