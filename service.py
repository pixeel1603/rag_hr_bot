import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, BSHTMLLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from confluence import get_page
from langchain_community.vectorstores.utils import filter_complex_metadata




os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
local_llm = "qwen2"


DOCUMENTS = [
    "docs/faq.pdf",
    "docs/projects.pdf"
]
Pages_id = [
    "2219540605",
    "2219540578",
    "2219540584",
    "2219540589",
    "2219540586",
    "2219540580",
    "2219540599",
    "2219540593",
    "2219540595",
    "2219540591",
    "2219540582",
    "2219540597"
]

def index():
    # docs = [PyPDFLoader(document).load_and_split() for document in DOCUMENTS]
    docs = [UnstructuredHTMLLoader(get_page(page), mode="paged").load() for page in Pages_id]
    docs_list = [item for sublist in docs for item in sublist]
    filtered_docs = filter_complex_metadata(docs_list)

    vectorstore = Chroma.from_documents(
        documents=filtered_docs,
        collection_name="rag-chroma",
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )
    retriever = vectorstore.as_retriever()
    return retriever

def retreval_grader(retriever, question) -> dict:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate


    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    return retrieval_grader.invoke({"question": question, "document": doc_txt})

def hallucination_grader(clue, generation) -> dict:
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        only single key 'score' and no preamble or explanation <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader.invoke({"documents": clue, "generation": generation})

def answer_grader(question, generation) -> dict:
    # LLM
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader.invoke({"question": question, "generation": generation})

def generate(retriever, question, history) -> (str, str):
    print('Start generating')
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Ты - HR-ассистент.
        Используй отрывки и историю чата ниже чтобы поддержать разговор. Если ты не знаешь, что ответь - просто скажи "Я уточню этот вопрос". Если в сообщении нет вопроса - просто отвечай как ответил бы человек
        Используй максиум 3 предложения и старайся сделать ответ как можно короче. Отвечай на русском языке <|eot_id|><|start_header_id|>user<|end_header_id|>
        Вопрос: {question}
        Context: {context}
        Chat history: {history}
        Ответ: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context", "history"],
    )

    llm = ChatOllama(model=local_llm, temperature=0.1)

    rag_chain = prompt | llm | StrOutputParser()

    print('Receiving docs')
    docs = retriever.invoke(question)
    if len(docs) > 3:
        clue = ["".join(docs[i].page_content) for i in range(3)]
    else:
        clue = ["".join(docs[i].page_content) for i in range(len(docs))]
    print('Call LLM')
    return rag_chain.invoke({"context": clue, "question": question, "history":history}), clue
