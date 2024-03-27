import streamlit as st

from utils import convert_pdf_to_images
from test_pdf import answer_question, get_retriever

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
    uploaded_file = st.file_uploader(
        "Choose a file", on_change=on_change, label_visibility="hidden"
    )


if uploaded_file and st.session_state.file_loaded:
    st.session_state.messages = []
    with st.spinner("Plik jest przetwarzany..."):
        text = convert_pdf_to_images(uploaded_file)
    st.session_state.context = text

    st.session_state.retriever = get_retriever(text)

    with st.empty():
        with st.chat_message("assistant"):
            response = st.write_stream(
                answer_question(
                    st.session_state.retriever,
                    "O czym jest ten plik? Opisz w 3/4 zdaniach.",
                )
            )
        st.write("")
    st.session_state.messages.append({"role": "ai", "content": response})
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
        message = st.write_stream(answer_question(st.session_state.retriever, prompt))
    st.session_state.messages.append({"role": "ai", "content": message})
