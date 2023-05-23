import os
import streamlit as st
from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor,PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
from pathlib import Path

os.environ["OPENAI_API_KEY"] = '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'
doc_path = './storage/'
index_file = 'index.json'

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
    
def send_click():
    query_engine = index.as_query_engine()
    st.session_state.response  = query_engine.query(st.session_state.prompt)

if check_password():
    if 'response' not in st.session_state:
        st.session_state.response = ''

    index = None
    st.title("Moxie Properties Chatbot")

    sidebar_placeholder = st.sidebar.container()
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:

        doc_files = os.listdir(doc_path)
        for doc_file in doc_files:
            os.remove(doc_path + doc_file)


        bytes_data = uploaded_file.read()
        with open(f"{doc_path}{uploaded_file.name}", 'wb') as f: 
            f.write(bytes_data)

        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            PandasExcelReader = download_loader("PandasExcelReader")
            loader = PandasExcelReader()
            documents = loader.load_data(uploaded_file)
        else:
            SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
            loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
            documents = loader.load_data()
        
        sidebar_placeholder.header('Current Processing Document:')
        sidebar_placeholder.subheader(uploaded_file.name)
        sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

        max_input_size = 4096
        num_output = 256
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        index.storage_context.persist()

    elif os.path.exists(index_file):
        storage_context = StorageContext.from_defaults(persist_dir="./storage")

        index = load_index_from_storage(storage_context)

        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
        documents = loader.load_data()
        doc_filename = os.listdir(doc_path)[0]
        sidebar_placeholder.header('Current Processing Document:')
        sidebar_placeholder.subheader(doc_filename)
        sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    if index != None:
        st.text_input("Ask something: ", key='prompt')
        st.button("Send", on_click=send_click)
        if st.session_state.response:
            st.subheader("Response: ")
            st.success(st.session_state.response, icon= "🤖")







