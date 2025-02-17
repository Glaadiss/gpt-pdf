import streamlit as st

from utils import convert_document_to_text
from test_pdf import answer_question, get_retriever, get_text_chunks_langchain

st.title("Podsumowanie pliku PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False


def on_change():
    st.session_state.file_loaded = True
    st.session_state.messages = []


with st.sidebar:
    uploaded_files = st.file_uploader(
        "Choose a file",
        on_change=on_change,
        label_visibility="hidden",
        accept_multiple_files=True,
    )


if uploaded_files and st.session_state.file_loaded:
    st.session_state.messages = []
    all_docs = []
    all_summaries = []
    with st.spinner("Indeksuje pliki..."):
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split(".")[-1]
            text = convert_document_to_text(uploaded_file, file_type)
            docs, summaries = get_text_chunks_langchain(text, uploaded_file.name)
            all_docs.extend(docs)
            all_summaries.extend(summaries)

        st.session_state.retriever = get_retriever(all_docs)
        st.session_state.retriever_summary = get_retriever(all_summaries)
    # with st.empty():
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(
    #             answer_question(
    #                 st.session_state.retriever,
    #                 "O czym jest ten plik? Opisz w 3/4 zdaniach.",
    #             )
    #         )
    #     st.write("")
    # st.session_state.messages.append({"role": "ai", "content": response})
    st.session_state.messages.append(
        {"role": "ai", "content": "O co chciałbyś zapytać?"}
    )

    st.session_state.file_loaded = False

# # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Zadaj pytanie odnośnie pliku PDF"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        streamer, meta = answer_question(
            st.session_state.retriever,
            prompt,
            retriever_summaries=st.session_state.retriever_summary,
            stream=True,
        )
        src = f"**Źródło: {meta['source']}** \n\n" if meta["source"] else ""
        st.write(src)
        message = st.write_stream(streamer)
    message = src + message
    st.session_state.messages.append({"role": "ai", "content": message})
