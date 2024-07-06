import os
import time
import datetime

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor

API_TOKEN = ''

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Initialize the retriever
history = {}

def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def init_engine():
    documents = SimpleDirectoryReader(
        input_files=["./docs/HR database.pdf", "./docs/Vacancies.pdf" ]
    ).load_data()
    document = Document(text="\n\n".join([doc.text for doc in documents]))

    index = build_sentence_window_index(
        [document],
        # llm=Ollama(model="llama3", request_timeout=120.0, base_url="http://92.53.65.80:8001"),
        llm=Ollama(model="llama3", request_timeout=120.0, base_url="http://127.0.0.1:11434"),
        save_dir="./sentence_index",
    )
    query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
    return query_engine

engine = init_engine()


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Welcome! Ask me anything.")


@dp.message_handler()
async def handle_message(message: types.Message):
    question = message.text

    if message.chat.id in history:
        history[message.chat.id] += " " + str(datetime.datetime.now()) + ":" + question
    else:
        history[message.chat.id] = question

    time.sleep(3)
    print("got request")
    await bot.send_chat_action(message.chat.id, action='typing')

    qa_template = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> Ты - HR-ассистент.
            Используй отрывки и историю чата ниже чтобы поддержать разговор. Если ты не знаешь, что ответь - просто скажи "Я уточню этот вопрос". Если в сообщении нет вопроса - просто отвечай как ответил бы человек
            Используй максиум 3 предложения и старайся сделать ответ как можно короче. Отвечай на русском языке <|eot_id|><|start_header_id|>user<|end_header_id|>
            Вопрос: {question}
            Chat history: {history}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = qa_template.format(question=question, history=history)

    report = engine.query(prompt)
    print(report)
    await message.reply(report)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)


