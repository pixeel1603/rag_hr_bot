import time
import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
from service import index, generate

API_TOKEN = ''

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Initialize the retriever
retriever = index()
history = {}


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

    generation, clue = generate(retriever, question, history[message.chat.id])

    await bot.send_chat_action(message.chat.id, action='typing')

    time.sleep(3)

    # hallucination_score = hallucination_grader(clue, generation)['score']
    # answer_score = answer_grader(question, generation)['score']
    #
    # if hallucination_score == "no" or answer_score == "no":
    #     await bot.send_chat_action(message.chat.id, action='typing')
    #     generation, clue = generate(retriever, question)

    print(generation)

    await message.reply(generation)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
