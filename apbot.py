import logging, os
from pathlib import Path
from turtle import update
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    filters,
    MessageHandler,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends import LocalShellBackend
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
import html
import json
import traceback
import models
import socket
import sys

load_dotenv()  # Load environment variables from .env file
CHAT_ID = int(os.environ["CHAT_ID"])
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
USER = os.environ["USER"]

telegram_bot = None

# llama.cpp
llm_model_gguf = ChatOpenAI(
    model="unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL",
    api_key="none",
    base_url="http://localhost:8080/v1",
    use_responses_api=False,
)

#http://localhost:8080/v1/models

# mlx
llm_model_mlx = ChatOpenAI(
    model=str(Path.home()) + "/Models/mlx-community/Qwen3.5-9B-MLX-4bit",
    api_key="none",
    base_url="http://localhost:8080/v1",
    use_responses_api=False,
)

# nvidia
nvidia_model = models.get_optimal_nvidia_model(NVIDIA_API_KEY)
llm_model_nvidia = ChatOpenAI(
    model=nvidia_model,
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    use_responses_api=False,
    stream_usage=False,
    max_retries=3,
    top_p=0.95,
    max_completion_tokens=16384 * 2,
)


# demo agent tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# web search tool
web_search = TavilySearch(max_results=5, topic="general")

# send file tool
async def send_file(filename: str) -> str:
    """Send a file or document as a Telegram message attachment."""
    if telegram_bot is None:
        raise RuntimeError("Telegram bot is not initialized.")
    await telegram_bot.send_document(chat_id=CHAT_ID, document=filename)
    return f"Sent File: {filename}"


# agent
print(f"Initializing Agent: {nvidia_model}")
memory_check_pointer = MemorySaver()
agent = create_deep_agent(
    model=llm_model_nvidia,
    tools=[send_file, web_search],
    backend=LocalShellBackend(
        root_dir=".", env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin"}, virtual_mode=True
    ),
    system_prompt=f"""
    "You are a helpful assistant (named: Agentic Pacific Bot) running on the {nvidia_model} AI model, that operates in a Telegram chat, specific to a user named {USER}."
    "You operate in a uv Python virtual environment, and can write and execute Python code."
    "You have the following libraries available: pypdf, python-pptx, trafilatura (to retrieve and extract web content), and can install additional libraries using `uv add <library_name>`."
    "Always generate files in the current directory and use relative paths. Never use absolute paths."
    "You can use the following tools: send_file(filename) to send a file or document to the user."
    "Always use the send_file tool to send files to the user instead of printing file contents.
    Use web search tool to find accurate, up-to-date information.
    """,
    store=InMemoryStore(),
    checkpointer=memory_check_pointer,
    # memory=["/AGENTS.md"],
)

# logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET/POST requests annoying logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# direct telegram command handlers
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a Telegram Bot for Agentic Pacific."
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id == CHAT_ID:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Agentic Pacific Bot - {nvidia_model} AI model @ {socket.gethostname()}",
        )


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id == CHAT_ID:
        message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Restarting @ {socket.gethostname()}",
        )
        print(message)
        os.execv(sys.executable, [sys.executable] + sys.argv)
        sys.exit(0)


# error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    # Log the error first
    logger.error("Exception: ", exc_info=context.error)

    # Get error class and message
    error_class = context.error.__class__.__name__
    error_message = str(context.error)
    error_text = f"<b>Error:</b> {error_class}\n<b>Message:</b> {error_message}"

    await context.bot.send_message(
        chat_id=CHAT_ID,
        text=error_text,
        parse_mode=ParseMode.HTML,
    )

    # Get the traceback as a string
    # tb_list = traceback.format_exception(
    #    None, context.error, context.error.__traceback__
    # )
    # tb_string = "".join(tb_list)
    # Build the message with some markup and additional information
    # update_str = update.to_dict() if isinstance(update, Update) else str(update)
    # message = (
    #    f"An exception was raised while handling an update\n\n"
    #    f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}</pre>\n\n"
    #    f"<pre>traceback =\n{html.escape(tb_string)}</pre>"
    # )
    # print(update_str)
    # print(tb_string)
    # You might want to send this message to a specific developer chat ID
    # await context.bot.send_message(
    #    chat_id=CHAT_ID, text=message, parse_mode=ParseMode.HTML
    # )


# handler to download attachments (photos/documents) sent by the Telegram user
async def download_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Identify if it's a Photo or a Document
    if update.message.photo:
        # It's a compressed photo.
        # Get the highest resolution version (last item in the list)
        attachment = update.message.photo[-1]
        # Photos don't have a 'file_name' attribute, so we create one
        filename = f"{attachment.file_unique_id}.jpg"
        print(f"Downloading Photo: {filename}")

    elif update.message.document:
        # It's an uncompressed file/document
        attachment = update.message.document
        # Documents DO have the original filename
        filename = attachment.file_name
        print(f"Downloading Document: {filename}")

    else:
        return  # Not an image

    # 2. Download the file
    new_file = await attachment.get_file()

    # Ensure a directory exists
    os.makedirs("downloads", exist_ok=True)
    save_path = os.path.join("downloads", filename)

    # 3. Save to disk
    await new_file.download_to_drive(custom_path=save_path)
    await update.message.reply_text(f"Recieved: {filename}")


# main message handler to process incoming messages and respond using the agent
async def process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if (
        update.effective_chat.id == CHAT_ID
    ):  # Only respond to messages from this chat ID (Sachin)
        # Run the agent
        print(f"Req Recieved: {update.message.text}")

        # download any files sent by the user
        await download_attachment(update, context)

        user_text = update.message.text or update.message.caption or "[file received]"
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config={"configurable": {"thread_id": "user_session_1"}},
        )
        # print(result)
        print(result["messages"][-1].content)
        reply = result["messages"][-1].content

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            reply_to_message_id=update.message.message_id,
            text=reply,
        )


if __name__ == "__main__":
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_bot = application.bot

    # handlers
    process_handler = MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.Document.ALL) & (~filters.COMMAND),
        process,
    )
    ping_handler = CommandHandler("ping", ping)
    restart_handler = CommandHandler("restart", restart)
    help_handler = CommandHandler("help", help)

    # register handlers
    application.add_handler(ping_handler)
    application.add_handler(restart_handler)
    application.add_handler(help_handler)
    application.add_handler(process_handler)

    # Register the error handler
    application.add_error_handler(error_handler)

    # run bot
    application.run_polling()
