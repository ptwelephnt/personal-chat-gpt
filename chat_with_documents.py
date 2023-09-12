import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from link_scraper import get_links, construct_full_links


def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader as file_loader
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader as file_loader
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader as file_loader
    else:
        print('Document format is not supported')
        return None

    print(f'Loading {file}')
    loader = file_loader(file)
    data = loader.load()
    return data


def load_urls():
    from langchain.document_loaders import SeleniumURLLoader

    urls = st.session_state.urls
    loader = SeleniumURLLoader(urls=urls)

    data = loader.load()
    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(model, vector_store, q, k=3, t=1):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=t)

    retriever = vector_store.as_retriever(serach_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(q)
    return answer


def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0001


def clear_text():
    st.session_state['clear_text'] = True


def clear_history():
    if 'history' in st.session_state:
        st.session_state.history = ''
    clear_text()


def add_url(url):
    if url not in st.session_state.urls:
        st.session_state.urls.append(url)


def remove_url(url):
    if url in st.session_state.urls:
        st.session_state.urls.remove(url)

def remove_all_urls():
    if 'urls' in st.session_state:
        del st.session_state.urls


if __name__ == '__main__':
    import os


    st.image('img.png')
    st.subheader('LLM Question-Answering Application')
    ### SIDEBAR ###
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        uploaded_file = False

        ### RADIO TO CHOOSE TO LOAD FILE OR URL(S) ###

        RADIO_CHOICES = ['File', 'URL']
        input_radio = st.radio('Load document(s) from...', RADIO_CHOICES)
        if input_radio == RADIO_CHOICES[0]:
            uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        elif input_radio == RADIO_CHOICES[1]:
            if 'urls' not in st.session_state:
                st.session_state['urls'] = []

            ### FORM TO UPLOAD URLS ###

            with st.form(key='url_form', clear_on_submit=True):
                url_input = st.text_input(label='URL:', key='url_input', value='', label_visibility='collapsed', placeholder='Enter URL')            
                scrape_url_checkbox = st.checkbox('Add URLs contained in page?')
                submitted = st.form_submit_button('Add')
                if submitted and url_input and not scrape_url_checkbox:
                    st.session_state.urls.append(url_input)
                elif submitted and url_input and scrape_url_checkbox:
                    scraped_links = get_links(url_input)
                    full_links = construct_full_links(scraped_links)
                    st.session_state.urls += full_links
            if st.session_state.urls:
                'Click url to remove:'
                i = 0
                for url in st.session_state.urls:
                    url_button = st.button(key=i ,label=url, on_click=remove_url, args=[url])
                    i += 1
                clear_urls_button = st.button('Clear all URLs', on_click=remove_all_urls)
        HELP_DICT = {
            'model': 'https://platform.openai.com/docs/models/overview',
            'chunk': 'https://www.pinecone.io/learn/chunking-strategies/',
            'k': 'Sample from "k" most probable tokens.',
            'temp': '''
                0 (More focused and coherent)
                to
                1 (More creative, but potentially less coherent)
                '''
        }
        MODEL_TYPES = ['gpt-3.5-turbo', 'gpt-4']
        model = st.selectbox('Model', MODEL_TYPES, index=0, on_change=clear_history, help=HELP_DICT['model'])
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=512, on_change=clear_history, help=HELP_DICT['chunk'])
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history, help=HELP_DICT['k'])
        temp = st.slider('Temperature', min_value=0.00, max_value=1.00, step=0.01, help=HELP_DICT['temp'])
        add_data = st.button('Add data', on_click=clear_history)

        if input_radio == RADIO_CHOICES[0] and uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully!')
        elif input_radio == RADIO_CHOICES[1] and len(st.session_state.urls) != 0 and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):

                data = load_urls()
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully!')

    placeholder = st.empty()
    q = placeholder.text_input('Ask a question about the content of your file:')
    if 'clear_text' in st.session_state:
        q = placeholder.text_input('Ask a question about the content of our file', value='', key='q_input')

    if q:
        if 'history' not in st.session_state:
            st.session_state['history'] = ''
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(model, vector_store, q, k)
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', key='history', height=400)
