import os
import re

# import dotenv
import openai
from langchain_openai import OpenAI
from utils import convert_document_to_text
import streamlit as st

# dotenv.load_dotenv()
# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
import os

# import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS


def remove_redundant_tabs_newlines(text):
    # Replace multiple tabs with a single tab
    text = re.sub(r"\t+", "\t", text)
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)
    return text


def format_docs(docs, n=20000):
    formatted = "\n\n".join(
        [remove_redundant_tabs_newlines(d.page_content) for d in docs]
    )

    return formatted[:n]


def get_text_chunks_langchain(text, name):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = [
        Document(page_content=x, metadata={"document_name": name})
        for x in text_splitter.split_text(text)
    ]

    retriever = get_retriever(docs)

    prompt = "kogo i czego dotyczy umowa (z uwzględniem miejsc, imion i nazwisk)"
    summary, meta = answer_question(retriever, prompt)

    summary_docs = [
        Document(page_content=summary.content, metadata={"document_name": name})
    ]

    return docs, summary_docs


def get_retriever(docs):
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever


documents_questions = {
    "example.pdf": [
        'Podaj dane osoby kontaktowej ze strony najemcy do umowy z "Rozalski Sp. z o.o. Sp. K" na obiekcie "Solna"?',
        'Czego dotyczy umowa najmu z "Rozalski Sp. z o.o. Sp. K" na obiekcie "Solna"??',
    ],
    # "example2.pdf": [
    #     'Do kiedy obowiązuje umowa najmu z najemcą "Lotto" na obiekcie "Zakopianka"?',
    #     'Ile wynosi czynsz za powierzchnię podstawową z najemcą "Lotto" na obiekcie "Zakopianka"?',
    # ],
    # "example3.pdf": [
    #     'Kiedy następuje przekazanie przedmiotu najmu dla najemcy w umowie najmu z "Bartosz Pieczykolan" na obiekcie "Zamość"?',
    #     'Podaj dane najemcy z  umowy najmu z "Bartosz Pieczykolan" na obiekcie "Zamość"',
    # ],
}

# model_name = "gpt-4-1106-preview"


model_name = "gpt-3.5-turbo-0125"


def get_stream(prompt):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = ChatOpenAI(model_name=model_name)
    return model.stream(prompt)
    # for chunk in model.stream(prompt):
    #     chunk.content
    #     print(chunk.content, end="", flush=True)
    # print("done")


def answer_question(retriever, question, retriever_summaries=None, stream=False):
    template = """Odpowiedz na pytanie bazując tylko na następującym kontekście używając tego samego języka, w którym zadano pytanie:
    {context}


    Pytanie: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model_name=model_name)

    def filter_docs(docs):

        docs_name = [d.metadata["document_name"] for d in docs]
        most_common_docs_name = docs_name[0]
        if retriever_summaries:
            summaries = retriever_summaries.get_relevant_documents(question)
            most_common_docs_name = summaries[0].metadata["document_name"]
        # most_common_docs_name = max(set(docs_name), key=docs_name.count)
        meta = {"source": ""}
        meta["source"] = most_common_docs_name
        filtered_docs = [
            d for d in docs if d.metadata["document_name"] == most_common_docs_name
        ]
        return filtered_docs, meta

    # model = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    ctx, meta = filter_docs(retriever.get_relevant_documents(question))

    formatted_docs = format_docs(ctx)

    prompt_executed = prompt.invoke({"context": formatted_docs, "question": question})
    if stream:
        model_result = model.stream(prompt_executed)
    else:
        model_result = model.invoke(prompt_executed)

    # chain = (
    #     {
    #         "context": Wrapper() | format_docs,
    #         "question": RunnablePassthrough(),
    #     }
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )

    answer = model_result
    return answer, meta


def answer_questions_for_pdf(pdf_path, questions):

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pdf_text = convert_document_to_text(pdf_path)
    docs = get_text_chunks_langchain(pdf_text)

    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    template = """Odpowiedz na pytanie bazując tylko na następującym kontekście używając tego samego języka, w którym zadano pytanie:
    {context}


    Pytanie: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model_name=model_name)

    def remove_redundant_tabs_newlines(text):
        # Replace multiple tabs with a single tab
        text = re.sub(r"\t+", "\t", text)
        # Replace multiple newlines with a single newline
        text = re.sub(r"\n+", "\n", text)
        return text

    def format_docs(docs):
        formatted = "\n\n".join(
            [remove_redundant_tabs_newlines(d.page_content) for d in docs]
        )

        return formatted[:20000]

    # model = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    for question in questions:
        answer = chain.invoke(question)
        print(f"Pytanie: {question}\nOdpowiedź: {answer} \n\n")


# for pdf, questions in documents_questions.items():
#     answer_questions_for_pdf(pdf, questions)
