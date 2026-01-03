import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import CommandStart

from config import BOT_TOKEN
from rag_core import answer

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(msg: Message):
    await msg.answer(
        "RAG bot is ready.\nAsk a question about the knowledge base."
    )


@dp.message()
async def handle_question(msg: Message):
    await msg.answer("Thinking...")
    try:
        resp = answer(msg.text)
    except Exception as e:
        resp = f"Error: {e}"

    await msg.answer(resp)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
