import os
import re

import dotenv
import openai
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from utils import convert_pdf_to_images

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

import os

import dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def get_retriever(text):
    docs = get_text_chunks_langchain(text)
    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
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


def answer_question(retriever, question):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    answer = chain.stream(question)
    return answer


def answer_questions_for_pdf(pdf_path, questions):

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    pdf_text = convert_pdf_to_images(pdf_path)
    docs = get_text_chunks_langchain(pdf_text)

    vectorstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    template = """Odpowiedz na pytanie bazując tylko na następującym kontekście używając tego samego języka, w którym zadano pytanie:
    {context}


    Pytanie: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125")

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
